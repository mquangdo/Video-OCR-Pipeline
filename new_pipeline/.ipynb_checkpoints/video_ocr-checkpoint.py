#!/usr/bin/env python3
"""
Video Subtitle OCR Pipeline
- Download YouTube video via yt-dlp
- Extract frames with OpenCV (no FFmpeg / sudo needed)
- Crop subtitle region (bottom 28%)
- Deduplicate frames via perceptual hash
- OCR with Qwen 2.5 VL via vLLM (OpenAI-compatible API)
- Output JSON with timestamp + text segments
"""

import os
import json
import base64
import shutil
import subprocess
import argparse
import logging
import tempfile
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import requests
from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://192.168.1.100:8000")
VLLM_MODEL    = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY", "token-abc123")  # vLLM default

# Fraction of frame height to crop as subtitle region (bottom portion)
SUBTITLE_REGION_FRACTION = 0.28

# Frame extraction rate (frames per second)
EXTRACT_FPS = 1.0

# Perceptual hash distance threshold (0 = identical, higher = more tolerant)
HASH_THRESHOLD = 4

# Minimum segment duration in seconds to include in output
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
    timestamp: float          # seconds
    path: Path
    crop_path: Optional[Path] = None
    text: str = ""
    phash: Optional[str] = None

@dataclass
class Segment:
    start: float
    end: float
    text: str


# ---------------------------------------------------------------------------
# Step 1 – Download video
# ---------------------------------------------------------------------------

def download_video(url: str, output_dir: Path) -> Path:
    """Download best video (no audio needed) via yt-dlp."""
    log.info("Downloading video: %s", url)
    out_template = str(output_dir / "video.%(ext)s")
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[ext=mp4]/bestvideo/best",
        "--output", out_template,
        "--no-playlist",
        url,
    ]
    subprocess.run(cmd, check=True)
    # Find downloaded file
    candidates = list(output_dir.glob("video.*"))
    if not candidates:
        raise FileNotFoundError("yt-dlp did not produce a video file")
    video_path = candidates[0]
    log.info("Downloaded: %s", video_path)
    return video_path


# ---------------------------------------------------------------------------
# Step 2 – Extract frames with OpenCV (no FFmpeg / sudo needed)
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, frames_dir: Path, fps: float = EXTRACT_FPS) -> list[Frame]:
    """
    Extract frames at the given FPS using OpenCV.
    Saves each sampled frame as JPEG and returns a list of Frame objects
    with accurate timestamps derived from the video's own PTS.
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    log.info("Extracting frames at %.2f fps via OpenCV ...", fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    log.info("Video: %.1fs, %.2f fps, %d total frames", duration, video_fps, total_frames)

    # How many source frames to skip between each sample
    frame_interval = max(1, round(video_fps / fps))

    frames: list[Frame] = []
    frame_idx = 0
    saved = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Use actual PTS-based timestamp for accuracy
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp = round(timestamp_ms / 1000.0, 3)

            out_path = frames_dir / f"frame_{saved:06d}.jpg"
            # OpenCV reads BGR; convert to RGB for PIL compatibility
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img.save(out_path, "JPEG", quality=92)

            frames.append(Frame(timestamp=timestamp, path=out_path))
            saved += 1

        frame_idx += 1

    cap.release()
    log.info("Extracted %d frames (sampled 1 per %d source frames)", saved, frame_interval)
    return frames


# ---------------------------------------------------------------------------
# Step 3 – Crop subtitle region
# ---------------------------------------------------------------------------

def crop_subtitle_region(frames: list[Frame], crop_dir: Path,
                          fraction: float = SUBTITLE_REGION_FRACTION) -> None:
    """Crop bottom `fraction` of each frame. Sets crop_path on each Frame."""
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
    """Compute perceptual hash for each cropped frame."""
    for frame in frames:
        img = Image.open(frame.crop_path)
        frame.phash = str(imagehash.phash(img))


def group_by_unique_subtitle(frames: list[Frame],
                              threshold: int = HASH_THRESHOLD) -> list[list[Frame]]:
    """
    Group consecutive frames that share the same subtitle image (similar hash).
    Returns list of groups; each group = same subtitle segment.
    """
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
    """Send a cropped frame to Qwen VL for OCR. Returns extracted text."""
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
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}"
                        },
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
    """OCR the middle frame of each group and assign text to all frames."""
    session = requests.Session()
    total = len(groups)
    for i, group in enumerate(groups, 1):
        # Pick middle frame as representative
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
    """
    Merge consecutive groups with identical text into final segments.
    Each segment has start/end timestamp and the OCR'd text.
    """
    if not groups:
        return []

    segments: list[Segment] = []
    flat: list[tuple[float, float, str]] = []  # (start, end, text)

    for group in groups:
        start = group[0].timestamp
        end = group[-1].timestamp
        text = group[0].text.strip()
        flat.append((start, end, text))

    # Merge consecutive identical texts
    merged_start, merged_end, merged_text = flat[0]
    for start, end, text in flat[1:]:
        if text == merged_text:
            merged_end = end
        else:
            if merged_text and (merged_end - merged_start) >= min_duration:
                segments.append(Segment(
                    start=round(merged_start, 2),
                    end=round(merged_end, 2),
                    text=merged_text,
                ))
            merged_start, merged_end, merged_text = start, end, text

    # Last segment
    if merged_text and (merged_end - merged_start) >= min_duration:
        segments.append(Segment(
            start=round(merged_start, 2),
            end=round(merged_end, 2),
            text=merged_text,
        ))

    log.info("Built %d final segments", len(segments))
    return segments


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def segments_to_json(segments: list[Segment], output_path: Path) -> None:
    data = [
        {
            "start": s.start,
            "end": s.end,
            "duration": round(s.end - s.start, 2),
            "text": s.text,
        }
        for s in segments
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info("Saved %d segments → %s", len(segments), output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    youtube_url: str,
    output_json: str = "output.json",
    work_dir: Optional[str] = None,
    keep_temp: bool = False,
    fps: float = EXTRACT_FPS,
    subtitle_fraction: float = SUBTITLE_REGION_FRACTION,
    hash_threshold: int = HASH_THRESHOLD,
):
    tmp = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="ocr_"))
    tmp.mkdir(parents=True, exist_ok=True)
    log.info("Working directory: %s", tmp)

    try:
        # 1. Download
        video_path = download_video(youtube_url, tmp / "download")

        # 2. Extract frames
        frames = extract_frames(video_path, tmp / "frames", fps=fps)
        if not frames:
            raise RuntimeError("No frames extracted from video")

        # 3. Crop subtitle region
        crop_subtitle_region(frames, tmp / "crops", fraction=subtitle_fraction)

        # 4. Compute perceptual hashes & group
        compute_phashes(frames)
        groups = group_by_unique_subtitle(frames, threshold=hash_threshold)

        # 5. OCR
        ocr_groups(groups)

        # 6. Build segments
        segments = build_segments(groups)

        # 7. Save JSON
        segments_to_json(segments, Path(output_json))

        log.info("✅ Done! Output: %s", output_json)
        return segments

    finally:
        if not keep_temp:
            shutil.rmtree(tmp, ignore_errors=True)
            log.info("Cleaned up temp dir: %s", tmp)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YouTube Video Subtitle OCR Pipeline using Qwen 2.5 VL"
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
    parser.add_argument("--vllm-url", default=None,
                        help="vLLM server base URL (overrides VLLM_BASE_URL env var)")
    parser.add_argument("--model", default=None,
                        help="Model name (overrides VLLM_MODEL env var)")

    args = parser.parse_args()

    if args.vllm_url:
        global VLLM_BASE_URL
        VLLM_BASE_URL = args.vllm_url
    if args.model:
        global VLLM_MODEL
        VLLM_MODEL = args.model

    run_pipeline(
        youtube_url=args.url,
        output_json=args.output,
        work_dir=args.work_dir,
        keep_temp=args.keep_temp,
        fps=args.fps,
        subtitle_fraction=args.subtitle_fraction,
        hash_threshold=args.hash_threshold,
    )


if __name__ == "__main__":
    main()