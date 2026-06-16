#!/usr/bin/env python3
"""
Download một video YouTube về máy.
Sử dụng: python download_video.py <URL> [--audio-only] [--quality 720]
"""

import yt_dlp
import argparse
import sys
import os


def download_video(url: str, output_dir: str = ".", audio_only: bool = False, quality: str = "720"):
    """
    Download video hoặc audio từ YouTube.

    Args:
        url: URL video YouTube
        output_dir: Thư mục lưu file (mặc định: thư mục hiện tại)
        audio_only: Chỉ tải audio (MP3)
        quality: Chất lượng video (360, 480, 720, 1080)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Template tên file: "001_Tên video.mp4"
    outtmpl = os.path.join(output_dir, "%(title)s.%(ext)s")

    if audio_only:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": outtmpl,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
        }
    else:
        # Chọn video có độ phân giải tối đa theo quality, fallback xuống thấp hơn nếu không có
        ydl_opts = {
            "format": f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]/best",
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
        }

    # Progress callback
    def progress_hook(d):
        if d["status"] == "downloading":
            percent = d.get("_percent_str", "?%").strip()
            speed = d.get("_speed_str", "?").strip()
            eta = d.get("_eta_str", "?").strip()
            print(f"\r  Đang tải: {percent} | Tốc độ: {speed} | Còn lại: {eta}   ", end="", flush=True)
        elif d["status"] == "finished":
            print(f"\n  ✅ Tải xong: {d['filename']}")
        elif d["status"] == "error":
            print(f"\n  ❌ Lỗi khi tải!")

    ydl_opts["progress_hooks"] = [progress_hook]

    print(f"🎬 URL: {url}")
    print(f"📁 Lưu vào: {os.path.abspath(output_dir)}")
    print(f"🎵 Chế độ: {'Chỉ audio (MP3)' if audio_only else f'Video MP4 ({quality}p)'}")
    print()

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Lấy thông tin video trước
        info = ydl.extract_info(url, download=False)
        title = info.get("title", "Unknown")
        duration = info.get("duration", 0)
        minutes = duration // 60
        seconds = duration % 60
        print(f"📌 Tiêu đề : {title}")
        print(f"⏱️  Thời lượng: {minutes}:{seconds:02d}")
        print()

        # Bắt đầu tải
        ydl.download([url])

    print("\n🎉 Hoàn tất!")


def main():
    parser = argparse.ArgumentParser(description="Download video YouTube")
    parser.add_argument("url", help="URL video YouTube")
    parser.add_argument("-o", "--output", default="downloads", help="Thư mục lưu (mặc định: ./downloads)")
    parser.add_argument("--audio-only", action="store_true", help="Chỉ tải audio MP3")
    parser.add_argument("--quality", default="720", choices=["360", "480", "720", "1080"], help="Chất lượng video (mặc định: 720p)")

    args = parser.parse_args()

    download_video(
        url=args.url,
        output_dir=args.output,
        audio_only=args.audio_only,
        quality=args.quality,
    )


if __name__ == "__main__":
    main()