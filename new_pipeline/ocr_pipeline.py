#!/usr/bin/env python3
"""
Video Subtitle OCR Pipeline — Local only (no DB, no MinIO)

Input  : YouTube playlist URL + start/end video index
Output : crawled_data/
            <video_id>/
                segments.json
                wavs/
                    segment_0000_f12-f45.wav
                    segment_0001_f46-f89.wav
                    ...

Usage:
    python ocr_pipeline.py <playlist_url> --start 0 --end 5
    python ocr_pipeline.py <playlist_url> --start 2 --end 2   # chỉ 1 video
"""

import os
import json
import base64
import shutil
import subprocess
import argparse
import logging
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import requests
from PIL import Image
import imagehash
from pydub import AudioSegment

import crawl_db

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http:localhost:8000")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY", "token-abc123")

# Vùng crop subtitle, dạng tỉ lệ tương đối (x1, y1, x2, y2), mỗi giá trị trong [0, 1].
# (0, 0) là góc trên-trái, (1, 1) là góc dưới-phải của frame.
# Mặc định: dải ngang full chiều rộng, nằm ở 72%-100% chiều cao (tương đương crop 28% đáy cũ).
SUBTITLE_REGION           = (0.0, 0.72, 1.0, 1.0)
EXTRACT_FPS               = 1.0
HASH_THRESHOLD            = 5
MIN_SEGMENT_DURATION      = 0.5

OUTPUT_ROOT = Path("crawled_data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    timestamp: float
    frame_index: int
    path: Path
    crop_path: Optional[Path] = None
    text: str = ""
    phash: Optional[str] = None

@dataclass
class Segment:
    start: float
    end: float
    start_frame: int
    end_frame: int
    text: str
    audio_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Step 0 — Lấy danh sách video từ playlist (1 lần gọi duy nhất)
# ---------------------------------------------------------------------------

@dataclass
class VideoEntry:
    url: str
    video_id: str
    title: str  # Thêm field này
    playlist_index: int = 0


def get_playlist_entries(playlist_url: str) -> list[VideoEntry]:
    log.info("Fetching playlist metadata: %s", playlist_url)
    # Lấy cả title
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--print", "url",
        "--print", "id",
        "--print", "title",
        "--no-warnings",
        playlist_url,
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]

    if len(lines) % 3 != 0:
        raise RuntimeError("yt-dlp không trả về đủ thông tin (url, id, title)")

    entries = []
    for i in range(0, len(lines), 3):
        # Hàm làm sạch tiêu đề để làm tên thư mục
        raw_title = lines[i + 2]
        clean_title = "".join([c if c.isalnum() or c in (" ", "-", "_") else "_" for c in raw_title])
        clean_title = clean_title.strip()[:100] # Giới hạn độ dài tránh lỗi đường dẫn

        entries.append(VideoEntry(
            url=lines[i],
            video_id=lines[i+1],
            title=clean_title,
            playlist_index=i // 3
        ))
    return entries


# ---------------------------------------------------------------------------
# Step 1 — Download video
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path) -> Path:
    log.info("Downloading: %s", url)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", out_template,
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True)
    candidates = list(output_dir.glob("video.*"))
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce a video file")
    video_path = candidates[0]
    log.info("Downloaded: %s", video_path)
    return video_path


# ---------------------------------------------------------------------------
# Step 2 — Extract frames
# ---------------------------------------------------------------------------

def get_video_fps(video_path: Path) -> float:
    """Lấy fps thật của video (dùng lại ở bước cắt audio theo frame index)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps


def extract_frames_generator(video_path: Path, frames_dir: Path, fps: float = EXTRACT_FPS):
    """Generator trích xuất từng frame một."""
    # Đảm bảo thư mục frames tồn tại
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(video_fps / fps))

    frame_idx = 0
    saved = 0
    while True:
        ret, bgr = cap.read()
        if not ret: break

        if frame_idx % frame_interval == 0:
            timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 3)
            # SỬ DỤNG frames_dir ĐƯỢC TRUYỀN VÀO
            out_path = frames_dir / f"frame_{saved:06d}.jpg"
            Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(out_path, "JPEG", quality=92)

            yield Frame(timestamp=timestamp, frame_index=frame_idx, path=out_path)
            saved += 1
        frame_idx += 1
    cap.release()


# ---------------------------------------------------------------------------
# Step 3 — Crop subtitle region
# ---------------------------------------------------------------------------

def crop_subtitle_region(frames: list[Frame], crop_dir: Path,
                          region: tuple[float, float, float, float] = SUBTITLE_REGION) -> None:
    """
    Crop vùng subtitle theo box (x1, y1, x2, y2).

    Mỗi tọa độ là tỉ lệ tương đối trong [0, 1] so với chiều rộng/cao của frame:
        x1, x2 : tỉ lệ theo chiều ngang (trái -> phải)
        y1, y2 : tỉ lệ theo chiều dọc   (trên -> dưới)

    Ví dụ: region=(0.0, 0.72, 1.0, 1.0) nghĩa là lấy full chiều rộng,
    từ 72% đến 100% chiều cao (dải đáy của frame).
    """
    x1, y1, x2, y2 = region
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"region không hợp lệ: {region} (cần 0<=x1<x2<=1 và 0<=y1<y2<=1)")

    crop_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img  = Image.open(frame.path)
        w, h = img.size
        box = (
            int(w * x1),
            int(h * y1),
            int(w * x2),
            int(h * y2),
        )
        cropped = img.crop(box)
        crop_path = crop_dir / frame.path.name
        cropped.save(crop_path, "JPEG", quality=90)
        frame.crop_path = crop_path
    log.info("Cropped region %s for %d frames", region, len(frames))


# ---------------------------------------------------------------------------
# Step 4 — Perceptual hash deduplication
# ---------------------------------------------------------------------------

def compute_phashes(frames: list[Frame]) -> None:
    for frame in frames:
        frame.phash = str(imagehash.phash(Image.open(frame.crop_path)))


def group_by_unique_subtitle(frames: list[Frame],
                              threshold: int = HASH_THRESHOLD) -> list[list[Frame]]:
    if not frames:
        return []

    groups        = [[frames[0]]]
    current_hash  = imagehash.hex_to_hash(frames[0].phash)

    for frame in frames[1:]:
        h = imagehash.hex_to_hash(frame.phash)
        if (current_hash - h) <= threshold:
            groups[-1].append(frame)
        else:
            groups.append([frame])
            current_hash = h

    log.info("Grouped into %d unique subtitle segments", len(groups))
    return groups


# ---------------------------------------------------------------------------
# Step 5 — OCR với Qwen 2.5 VL
# ---------------------------------------------------------------------------

def ocr_frame(image_path: Path, session: requests.Session) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": VLLM_MODEL,
        "max_tokens": 256,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract ONLY the subtitle/caption text visible in this image. "
                        "Return the raw text exactly as shown, preserving line breaks. "
                        "If there is no text, return an empty string. "
                        "Do NOT add any explanation or punctuation that is not in the image."
                    ),
                },
            ],
        }],
    }

    resp = session.post(
        f"{VLLM_BASE_URL}/v1/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def ocr_groups(groups: list[list[Frame]]) -> None:
    session = requests.Session()
    total   = len(groups)
    for i, group in enumerate(groups, 1):
        rep = group[len(group) // 2]
        log.info("[%d/%d] OCR @ %.2fs ...", i, total, rep.timestamp)
        try:
            text = ocr_frame(rep.crop_path, session)
        except Exception as e:
            log.warning("OCR failed: %s", e)
            text = ""
        for frame in group:
            frame.text = text
        log.info("  → %r", text[:80])


# ---------------------------------------------------------------------------
# Step 6 — Build segments
# ---------------------------------------------------------------------------

def build_segments(groups: list[list[Frame]],
                   min_duration: float = MIN_SEGMENT_DURATION) -> list[Segment]:
    if not groups:
        return []

    flat = [
        (g[0].timestamp, g[-1].timestamp, g[0].frame_index, g[-1].frame_index, g[0].text.strip())
        for g in groups
    ]

    segments = []
    ms, me, mfs, mfe, mt = flat[0]

    for start_s, end_s, start_f, end_f, text in flat[1:]:
        if text == mt:
            me, mfe = end_s, end_f
        else:
            if mt and (me - ms) >= min_duration:
                segments.append(Segment(round(ms, 3), round(me, 3), mfs, mfe, mt))
            ms, me, mfs, mfe, mt = start_s, end_s, start_f, end_f, text

    if mt and (me - ms) >= min_duration:
        segments.append(Segment(round(ms, 3), round(me, 3), mfs, mfe, mt))

    log.info("Built %d segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Step 7 — Extract audio → wavs/
# ---------------------------------------------------------------------------

def extract_audio_segments(segments: list[Segment], video_path: Path,
                            wavs_dir: Path, video_fps: float) -> None:
    wavs_dir.mkdir(parents=True, exist_ok=True)

    try:
        full_audio = AudioSegment.from_file(str(video_path))
    except Exception as e:
        log.warning("Cannot read audio: %s — skipping", e)
        return

    log.info("Extracting %d audio clips ...", len(segments))
    for i, seg in enumerate(segments):
        start_ms = int(seg.start_frame / video_fps * 1000)
        end_ms   = min(int(seg.end_frame / video_fps * 1000), len(full_audio))
        if start_ms >= end_ms:
            continue

        out_path = wavs_dir / f"segment_{i:04d}_f{seg.start_frame}-f{seg.end_frame}.wav"
        full_audio[start_ms:end_ms].export(str(out_path), format="wav")
        seg.audio_path = out_path

    log.info("Audio extraction complete → %s", wavs_dir)


# ---------------------------------------------------------------------------
# Step 8 — Lưu segments.json
# ---------------------------------------------------------------------------

def save_segments_json(segments: list[Segment], output_path: Path) -> None:
    data = [
        {
            "start":       s.start,
            "end":         s.end,
            "duration":    round(s.end - s.start, 3),
            "start_frame": s.start_frame,
            "end_frame":   s.end_frame,
            "text":        s.text,
            "audio_file":  s.audio_path.name if s.audio_path else None,
        }
        for s in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Saved %d segments → %s", len(data), output_path)


# ---------------------------------------------------------------------------
# Single video pipeline
# ---------------------------------------------------------------------------

def process_video(
    entry: "VideoEntry",
    video_out_dir: Path,
    playlist_url: Optional[str]                       = None,
    fps: float                                         = EXTRACT_FPS,
    subtitle_region: tuple[float, float, float, float] = SUBTITLE_REGION,
    hash_threshold: int                                = HASH_THRESHOLD,
    keep_temp: bool                                    = False,
) -> bool:
    """
    Xử lý 1 video. Trả về True nếu thành công, False nếu lỗi.

    Output layout:
        video_out_dir/
            segments.json
            wavs/
                segment_0000_*.wav
                ...
    """
    video_id    = entry.video_id
    youtube_url = entry.url

    # ── Idempotency: hỏi DB trước ────────────────────────────────────────
    row = crawl_db.create_or_get(
        video_id       = video_id,
        url            = youtube_url,
        playlist_url   = playlist_url,
        playlist_index = entry.playlist_index,
    )

    if row["status"] == "done":
        log.info("DB: status=done → bỏ qua %s", video_id)
        return True

    if row["status"] == "processing":
        # Crash lần trước giữa chừng → cho phép chạy lại
        log.warning("DB: status=processing (crash?) → xử lý lại %s", video_id)

    crawl_db.mark_processing(video_id)

    # Đặt tên folder theo SỐ THỨ TỰ trong playlist (1-based, 4 chữ số) + video_id
    # để giữ đúng thứ tự khi sort theo tên, đồng thời vẫn tránh trùng tên.
    folder_name = f"{entry.playlist_index + 1:04d}_{video_id}"
    video_out_dir = video_out_dir.parent / folder_name

    video_out_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = video_out_dir / "wavs"
    tmp      = Path(tempfile.mkdtemp(prefix="ocr_tmp_"))

    try:
        # 1. Download
        video_path = download_video(youtube_url, tmp / "download")

        # 2. Extract frames
        #    (extract_frames_generator là generator -> phải "tiêu thụ" thành list
        #     trước khi dùng len()/truthiness, nếu không sẽ lỗi NameError: frames)
        frames = list(extract_frames_generator(
            video_path=video_path,
            frames_dir=tmp / "frames",
            fps=fps
        ))
        if not frames:
            raise RuntimeError("No frames extracted")

        # fps thật của video, dùng để quy đổi frame_index -> ms ở bước cắt audio
        video_fps = get_video_fps(video_path)

        # 3. Crop
        crop_subtitle_region(frames, tmp / "crops", region=subtitle_region)

        # 4. Hash & group
        compute_phashes(frames)
        groups = group_by_unique_subtitle(frames, threshold=hash_threshold)

        # 5. OCR
        ocr_groups(groups)

        # 6. Build segments
        segments = build_segments(groups)

        # 7. Audio → wavs/
        extract_audio_segments(segments, video_path, wavs_dir, video_fps)

        # 8. Save JSON
        save_segments_json(segments, video_out_dir / "segments.json")

        # 9. Lưu metadata vào DB
        total_duration = sum(s.end - s.start for s in segments)
        crawl_db.mark_done(
            video_id       = video_id,
            output_dir     = str(video_out_dir.resolve()),
            segment_count  = len(segments),
            total_duration = round(total_duration, 3),
        )

        log.info("✅ Done: %s (%d segments, %.1fs)", video_id, len(segments), total_duration)
        return True

    except Exception as e:
        crawl_db.mark_failed(video_id, str(e))
        log.error("❌ Failed (%s): %s", video_id, e)
        shutil.rmtree(video_out_dir, ignore_errors=True)
        return False

    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main — xử lý playlist theo range
# ---------------------------------------------------------------------------

def run(
    playlist_url: str,
    entries: list[VideoEntry],
    start_index: int,
    end_index: int,
    output_root: Path                                  = OUTPUT_ROOT,
    fps: float                                          = EXTRACT_FPS,
    subtitle_region: tuple[float, float, float, float]  = SUBTITLE_REGION,
    hash_threshold: int                                 = HASH_THRESHOLD,
    keep_temp: bool                                     = False,
):
    if start_index < 0 or end_index >= len(entries) or start_index > end_index:
        raise ValueError(
            f"Index không hợp lệ: start={start_index}, end={end_index}, "
            f"playlist có {len(entries)} video (0–{len(entries)-1})"
        )

    selected = entries[start_index : end_index + 1]

    # ── Check trước: hỏi DB xem video nào đã done ───────────────────────
    todo: list[VideoEntry] = []
    skip: list[VideoEntry] = []

    for entry in selected:
        if crawl_db.is_done(entry.video_id):
            skip.append(entry)
        else:
            todo.append(entry)

    log.info("─" * 60)
    log.info("Playlist range [%d→%d]: %d video tổng", start_index, end_index, len(selected))
    log.info("  ✅ Đã xử lý (bỏ qua) : %d", len(skip))
    log.info("  🔜 Cần xử lý         : %d", len(todo))
    if skip:
        for e in skip:
            log.info("     skip  %s  (%s)", e.video_id, e.url)
    if not todo:
        log.info("Không có video nào cần xử lý.")
        return {"success": [e.url for e in skip], "skipped": [e.url for e in skip], "failed": []}

    # ── Xử lý từng video còn lại ─────────────────────────────────────────
    results = {
        "success": [e.url for e in skip],
        "skipped": [e.url for e in skip],
        "failed":  [],
    }

    for i, entry in enumerate(todo, 1):
        log.info("─" * 60)
        log.info("[%d/%d] %s  (id: %s)", i, len(todo), entry.url, entry.video_id)

        video_out_dir = output_root / entry.video_id
        ok = process_video(
            entry             = entry,
            video_out_dir     = video_out_dir,
            playlist_url      = playlist_url,
            fps               = fps,
            subtitle_region   = subtitle_region,
            hash_threshold    = hash_threshold,
            keep_temp         = keep_temp,
        )
        if ok:
            results["success"].append(entry.url)
        else:
            results["failed"].append(entry.url)

    # ── Summary ───────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("✅ Success (bao gồm skip) : %d", len(results["success"]))
    log.info("   └─ Skipped (đã có)    : %d", len(results["skipped"]))
    log.info("❌ Failed                 : %d", len(results["failed"]))
    if results["failed"]:
        for u in results["failed"]:
            log.info("   • %s", u)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Playlist OCR Pipeline — output local crawled_data/"
    )
    parser.add_argument("playlist_url",
                        help="URL của YouTube playlist")
    parser.add_argument("--start", type=int, default=0,
                        help="Index video bắt đầu (0-based, mặc định: 0)")
    parser.add_argument("--end", type=int, default=None,
                        help="Index video kết thúc (inclusive, mặc định: video cuối)")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT),
                        help=f"Thư mục output (mặc định: {OUTPUT_ROOT})")
    parser.add_argument("--fps", type=float, default=EXTRACT_FPS,
                        help=f"Frame extraction rate (mặc định: {EXTRACT_FPS})")
    parser.add_argument("--subtitle-region", type=float, nargs=4,
                        default=list(SUBTITLE_REGION),
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Vùng crop subtitle theo tỉ lệ tương đối x1 y1 x2 y2, "
                             f"mỗi giá trị trong [0,1] (mặc định: {SUBTITLE_REGION})")
    parser.add_argument("--hash-threshold", type=int, default=HASH_THRESHOLD,
                        help=f"Perceptual hash threshold (mặc định: {HASH_THRESHOLD})")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Giữ lại temp files sau khi xử lý")
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM server URL (override VLLM_BASE_URL)")
    parser.add_argument("--model", default=None,
                        help="Model name (override VLLM_MODEL)")

    args = parser.parse_args()

    if args.vllm_url:
        global VLLM_BASE_URL
        VLLM_BASE_URL = args.vllm_url
    if args.model:
        global VLLM_MODEL
        VLLM_MODEL = args.model

    # Khởi tạo schema DB (idempotent — an toàn khi gọi lại)
    crawl_db.init_schema()

    # Fetch playlist 1 lần để biết tổng số video → resolve --end
    entries   = get_playlist_entries(args.playlist_url)
    end_index = args.end if args.end is not None else len(entries) - 1

    run(
        playlist_url      = args.playlist_url,
        entries           = entries,
        start_index       = args.start,
        end_index         = end_index,
        output_root       = Path(args.output_dir),
        fps               = args.fps,
        subtitle_region   = tuple(args.subtitle_region),
        hash_threshold    = args.hash_threshold,
        keep_temp         = args.keep_temp,
    )


if __name__ == "__main__":
    main()