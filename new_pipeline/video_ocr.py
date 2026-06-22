#!/usr/bin/env python3
"""
Video Subtitle OCR Pipeline (+ PostgreSQL persistence)
- Download YouTube video (video + audio) via yt-dlp
- Extract frames with OpenCV (no FFmpeg / sudo needed)
- Crop subtitle region (bottom 28%)
- Deduplicate frames via perceptual hash
- OCR with Qwen 2.5 VL via vLLM (OpenAI-compatible API)
- Extract audio clips per segment with pydub
- Output JSON with timestamp + text + audio_file segments
- Persist video status + segments into PostgreSQL (idempotent re-runs)
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

import db            # module: db.py (PostgreSQL)
import minio_client  # module: minio_client.py (object storage cho audio)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.1.100:8000")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY", "token-abc123")  # vLLM default

SUBTITLE_REGION_FRACTION = 0.28
EXTRACT_FPS = 1.0
HASH_THRESHOLD = 4
MIN_SEGMENT_DURATION = 0.5

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
# Step 1 – Download video
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path) -> Path:
    log.info("Downloading video: %s", url)
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
# Step 2 – Extract frames with OpenCV
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path, fps: float = EXTRACT_FPS) -> tuple[list[Frame], float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting frames at %.2f fps via OpenCV ...", fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    log.info("Video: %.1fs, %.2f fps, %d total frames", duration, video_fps, total_frames)

    frame_interval = max(1, round(video_fps / fps))

    frames: list[Frame] = []
    frame_idx = 0
    saved = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = round(timestamp_ms / 1000.0, 3)

            out_path = frames_dir / f"frame_{saved:06d}.jpg"
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(out_path, "JPEG", quality=92)

            frames.append(Frame(timestamp=timestamp, frame_index=frame_idx, path=out_path))
            saved += 1

        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames (sampled 1 per %d source frames)", saved, frame_interval)
    return frames, video_fps


# ---------------------------------------------------------------------------
# Step 3 – Crop subtitle region
# ---------------------------------------------------------------------------

def crop_subtitle_region(frames: list[Frame], crop_dir: Path,
                          fraction: float = SUBTITLE_REGION_FRACTION) -> None:
    crop_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = Image.open(frame.path)
        w, h = img.size
        top = int(h * (1 - fraction))
        cropped = img.crop((0, top, w, h))
        crop_path = crop_dir / frame.path.name
        cropped.save(crop_path, "JPEG", quality=90)
        frame.crop_path = crop_path
    log.info("Cropped subtitle region (bottom %.0f%%) for %d frames",
             fraction * 100, len(frames))


# ---------------------------------------------------------------------------
# Step 4 – Perceptual hash deduplication
# ---------------------------------------------------------------------------

def compute_phashes(frames: list[Frame]) -> None:
    for frame in frames:
        img = Image.open(frame.crop_path)
        frame.phash = str(imagehash.phash(img))


def group_by_unique_subtitle(frames: list[Frame],
                              threshold: int = HASH_THRESHOLD) -> list[list[Frame]]:
    if not frames:
        return []

    groups: list[list[Frame]] = []
    current_group = [frames[0]]
    current_hash = imagehash.hex_to_hash(frames[0].phash)

    for frame in frames[1:]:
        h = imagehash.hex_to_hash(frame.phash)
        dist = current_hash - h
        if dist <= threshold:
            current_group.append(frame)
        else:
            groups.append(current_group)
            current_group = [frame]
            current_hash = h

    groups.append(current_group)
    log.info("Grouped into %d unique subtitle segments", len(groups))
    return groups


# ---------------------------------------------------------------------------
# Step 5 – OCR with Qwen 2.5 VL
# ---------------------------------------------------------------------------

def image_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def ocr_frame(image_path: Path, session: requests.Session) -> str:
    b64 = image_to_base64(image_path)
    payload = {
        "model": VLLM_MODEL,
        "max_tokens": 256,
        "temperature": 0.0,
        "messages": [
            {
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
            }
        ],
    }

    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    resp = session.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return text


def ocr_groups(groups: list[list[Frame]]) -> None:
    session = requests.Session()
    total = len(groups)
    for i, group in enumerate(groups, 1):
        rep_frame = group[len(group) // 2]
        log.info("[%d/%d] OCR frame @ %.2fs ...", i, total, rep_frame.timestamp)
        try:
            text = ocr_frame(rep_frame.crop_path, session)
        except Exception as e:
            log.warning("OCR failed for frame %s: %s", rep_frame.path.name, e)
            text = ""
        for frame in group:
            frame.text = text
        log.info("  → %r", text[:80])


# ---------------------------------------------------------------------------
# Step 6 – Build segments
# ---------------------------------------------------------------------------

def build_segments(groups: list[list[Frame]],
                   min_duration: float = MIN_SEGMENT_DURATION) -> list[Segment]:
    if not groups:
        return []

    segments: list[Segment] = []
    flat: list[tuple[float, float, int, int, str]] = []

    for group in groups:
        start_s = group[0].timestamp
        end_s   = group[-1].timestamp
        start_f = group[0].frame_index
        end_f   = group[-1].frame_index
        text    = group[0].text.strip()
        flat.append((start_s, end_s, start_f, end_f, text))

    ms, me, mfs, mfe, mt = flat[0]
    for start_s, end_s, start_f, end_f, text in flat[1:]:
        if text == mt:
            me  = end_s
            mfe = end_f
        else:
            if mt and (me - ms) >= min_duration:
                segments.append(Segment(
                    start=round(ms, 3), end=round(me, 3),
                    start_frame=mfs, end_frame=mfe,
                    text=mt,
                ))
            ms, me, mfs, mfe, mt = start_s, end_s, start_f, end_f, text

    if mt and (me - ms) >= min_duration:
        segments.append(Segment(
            start=round(ms, 3), end=round(me, 3),
            start_frame=mfs, end_frame=mfe,
            text=mt,
        ))

    log.info("Built %d final segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Step 7 – Extract audio clips per segment
# ---------------------------------------------------------------------------

def extract_audio_segments(segments: list[Segment], video_path: Path,
                            audio_dir: Path, video_fps: float,
                            video_id: Optional[int] = None,
                            use_minio: bool = True) -> None:
    """
    Cut audio using exact frame indices → milliseconds (frame_index / video_fps * 1000).
    Nếu use_minio=True: upload mỗi clip lên MinIO rồi xoá file local, seg.audio_path
    sẽ lưu object_name (key trong bucket) thay vì path local.
    Nếu upload lỗi (MinIO down...): tự fallback giữ file local, không làm crash pipeline.
    """
    audio_dir.mkdir(parents=True, exist_ok=True)
    log.info("Loading audio track from video ...")

    try:
        full_audio = AudioSegment.from_file(str(video_path))
    except Exception as e:
        log.warning("pydub could not read audio: %s — skipping audio extraction", e)
        return

    if use_minio:
        try:
            minio_client.ensure_bucket()
        except Exception as e:
            log.warning("Không kết nối được MinIO (%s) — toàn bộ audio sẽ lưu local", e)
            use_minio = False

    log.info("Extracting %d audio clips (video_fps=%.3f) ...", len(segments), video_fps)
    for i, seg in enumerate(segments):
        start_ms = int(seg.start_frame / video_fps * 1000)
        end_ms   = int(seg.end_frame   / video_fps * 1000)
        end_ms   = min(end_ms, len(full_audio))
        if start_ms >= end_ms:
            continue
        clip = full_audio[start_ms:end_ms]
        out_path = audio_dir / f"segment_{i:04d}_f{seg.start_frame}-f{seg.end_frame}.wav"
        clip.export(str(out_path), format="wav")

        if use_minio:
            prefix = f"video_{video_id}" if video_id is not None else "unknown_video"
            object_name = f"{prefix}/{out_path.name}"
            try:
                minio_client.upload_file(out_path, object_name)
                out_path.unlink()  # MinIO là nguồn lưu chính, xoá bản local để khỏi trùng
                seg.audio_path = object_name
            except Exception as e:
                log.warning("Upload MinIO thất bại cho %s: %s — giữ file local", out_path.name, e)
                seg.audio_path = out_path
        else:
            seg.audio_path = out_path

        log.info("  [%d] frame %d–%d (%.3fs–%.3fs) → %s",
                 i, seg.start_frame, seg.end_frame, seg.start, seg.end, seg.audio_path)

    log.info("Audio extraction complete")


def segments_to_json(segments: list[Segment], output_path: Path) -> list[dict]:
    """Lưu segments ra JSON, trả về list dict để dùng tiếp (insert DB)."""
    data = [
        {
            "start": s.start,
            "end": s.end,
            "duration": round(s.end - s.start, 3),
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "text": s.text,
            "audio_file": str(s.audio_path) if s.audio_path else None,
        }
        for s in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Saved %d segments → %s", len(segments), output_path)
    return data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    youtube_url: str,
    output_json: str = "output.json",
    audio_output_dir: Optional[str] = None,
    work_dir: Optional[str] = None,
    keep_temp: bool = False,
    fps: float = EXTRACT_FPS,
    subtitle_fraction: float = SUBTITLE_REGION_FRACTION,
    hash_threshold: int = HASH_THRESHOLD,
    force: bool = False,
    use_minio: bool = True,
):
    # --- DB: idempotency ---------------------------------------------------
    video_row = db.create_or_get_video(youtube_url)
    video_id = video_row["id"]

    if video_row["status"] == "done" and not force:
        log.info(
            "Video đã xử lý xong trước đó (id=%s, status=done) → bỏ qua. "
            "Dùng --force nếu muốn xử lý lại.",
            video_id,
        )
        return None

    if force and video_row["status"] != "pending":
        db.reset_video_for_reprocess(video_id)

    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ocr_"))
    tmp.mkdir(parents=True, exist_ok=True)
    log.info("Working directory: %s", tmp)

    audio_dir = Path(audio_output_dir) if audio_output_dir else Path(output_json).parent / "audio"

    try:
        # 1. Download
        db.update_video_status(video_id, "downloading")
        video_path = download_video(youtube_url, tmp / "download")
        db.update_video_status(video_id, "downloading", video_path=str(video_path))

        # 2. Extract frames
        db.update_video_status(video_id, "extracting")
        frames, video_fps = extract_frames(video_path, tmp / "frames", fps=fps)
        if not frames:
            raise RuntimeError("No frames extracted from video")
        duration = frames[-1].timestamp
        db.update_video_status(video_id, "extracting", video_fps=video_fps, duration=duration)

        # 3. Crop subtitle region
        crop_subtitle_region(frames, tmp / "crops", fraction=subtitle_fraction)

        # 4. Hash & group
        compute_phashes(frames)
        groups = group_by_unique_subtitle(frames, threshold=hash_threshold)

        # 5. OCR
        db.update_video_status(video_id, "ocr_processing")
        ocr_groups(groups)

        # 6. Build segments
        segments = build_segments(groups)

        # 7. Audio clips (trước khi cleanup!)
        db.update_video_status(video_id, "audio_extracting")
        extract_audio_segments(segments, video_path, audio_dir, video_fps,
                                video_id=video_id, use_minio=use_minio)

        # 8. Lưu JSON
        segments_data = segments_to_json(segments, Path(output_json))

        # --- DB: persist segments ------------------------------------------
        db.delete_segments_for_video(video_id)  # an toàn khi re-run
        db.insert_segments(video_id, segments_data)
        db.update_video_status(video_id, "done")

        log.info("✅ Done! Output: %s", output_json)
        log.info("🔊 Audio clips: %s", audio_dir)
        return segments

    except Exception as e:
        db.update_video_status(video_id, "failed", error_message=str(e))
        log.error("❌ Pipeline failed (video_id=%s): %s", video_id, e)
        raise

    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
            log.info("Cleaned up temp dir: %s", tmp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Subtitle OCR Pipeline using Qwen 2.5 VL (+ PostgreSQL)"
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("-o", "--output", default="output.json",
                        help="Output JSON file path (default: output.json)")
    parser.add_argument("--work-dir", default=None,
                        help="Working directory for temp files (default: system temp)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files after processing")
    parser.add_argument("--fps", type=float, default=EXTRACT_FPS,
                        help=f"Frame extraction rate (default: {EXTRACT_FPS})")
    parser.add_argument("--subtitle-fraction", type=float, default=SUBTITLE_REGION_FRACTION,
                        help=f"Bottom fraction of frame for subtitle crop (default: {SUBTITLE_REGION_FRACTION})")
    parser.add_argument("--hash-threshold", type=int, default=HASH_THRESHOLD,
                        help=f"Perceptual hash diff threshold (default: {HASH_THRESHOLD})")
    parser.add_argument("--audio-dir", default=None,
                        help="Directory to save audio clips (default: <output_dir>/audio/)")
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM server base URL (overrides VLLM_BASE_URL env var)")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides VLLM_MODEL env var)")
    parser.add_argument("--force", action="store_true",
                        help="Xử lý lại video dù đã có status='done' trong DB")
    parser.add_argument("--no-minio", action="store_true",
                        help="Không upload audio lên MinIO, chỉ lưu local disk")
    parser.add_argument("--no-db-init", action="store_true",
                        help="Bỏ qua bước tự tạo schema (dùng khi schema đã được setup sẵn)")

    args = parser.parse_args()

    if args.vllm_url:
        global VLLM_BASE_URL
        VLLM_BASE_URL = args.vllm_url
    if args.model:
        global VLLM_MODEL
        VLLM_MODEL = args.model

    if not args.no_db_init:
        db.init_schema()

    run_pipeline(
        youtube_url=args.url,
        output_json=args.output,
        audio_output_dir=args.audio_dir,
        work_dir=args.work_dir,
        keep_temp=args.keep_temp,
        fps=args.fps,
        subtitle_fraction=args.subtitle_fraction,
        hash_threshold=args.hash_threshold,
        force=args.force,
        use_minio=not args.no_minio,
    )


if __name__ == "__main__":
    main()