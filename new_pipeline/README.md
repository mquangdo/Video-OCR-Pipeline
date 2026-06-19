# Video Subtitle OCR Pipeline

Trích xuất phụ đề burned-in từ video YouTube bằng Qwen 2.5 VL chạy trên vLLM.

## Pipeline

```
YouTube URL
    │
    ▼
[yt-dlp] Download video
    │
    ▼
[FFmpeg] Extract frames @ 1fps
    │
    ▼
[PIL] Crop vùng phụ đề (bottom 28%)
    │
    ▼
[imagehash] Nhóm frame có cùng phụ đề (perceptual hash)
    │
    ▼
[Qwen 2.5 VL / vLLM] OCR 1 frame đại diện / nhóm
    │
    ▼
[Merge] Gộp các segment liên tiếp cùng text
    │
    ▼
output.json  { start, end, duration, text }
```

## Cài đặt

```bash
# 1. Cài ffmpeg (nếu chưa có)
sudo apt install ffmpeg          # Ubuntu/Debian
brew install ffmpeg              # macOS

# 2. Cài Python dependencies
pip install -r requirements.txt
```

## Cấu hình vLLM

Sửa biến môi trường hoặc dùng CLI flags:

```bash
export VLLM_BASE_URL="http://<IP_SERVER>:8000"
export VLLM_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
export VLLM_API_KEY="token-abc123"   # API key vLLM của bạn
```

## Sử dụng

```bash
# Cơ bản
python ocr_pipeline.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Chỉ định output file
python ocr_pipeline.py "https://youtu.be/VIDEO_ID" -o result.json

# Tùy chỉnh vLLM server
python ocr_pipeline.py "URL" --vllm-url http://192.168.1.50:8000

# Giữ lại temp files để debug
python ocr_pipeline.py "URL" --keep-temp --work-dir ./debug_output

# Tăng FPS nếu phụ đề thay đổi nhanh
python ocr_pipeline.py "URL" --fps 2

# Mở rộng vùng crop phụ đề (nếu phụ đề cao hơn)
python ocr_pipeline.py "URL" --subtitle-fraction 0.35
```

## Output JSON

```json
[
  {
    "start": 1.0,
    "end": 4.5,
    "duration": 3.5,
    "text": "mô hình đã giúp gia đình anh phát triển và có thu nhập ổn định"
  },
  {
    "start": 5.0,
    "end": 8.0,
    "duration": 3.0,
    "text": "trung bình từ 400-500 triệu đồng mỗi năm."
  }
]
```

## Tham số CLI

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `url` | (bắt buộc) | YouTube URL |
| `-o / --output` | `output.json` | File output |
| `--vllm-url` | env `VLLM_BASE_URL` | URL server vLLM |
| `--model` | env `VLLM_MODEL` | Tên model |
| `--fps` | `1.0` | Số frame/giây extract |
| `--subtitle-fraction` | `0.28` | Tỉ lệ crop phía dưới |
| `--hash-threshold` | `4` | Ngưỡng hash khác nhau (0=giống hệt) |
| `--work-dir` | system temp | Thư mục làm việc |
| `--keep-temp` | False | Giữ file tạm |

## Tuning gợi ý

- **Phụ đề thay đổi nhanh** (< 1s/câu): tăng `--fps 2` hoặc `--fps 3`
- **Crop bị thiếu text**: tăng `--subtitle-fraction 0.35`
- **Nhận nhầm 2 phụ đề khác nhau là 1**: giảm `--hash-threshold 2`
- **Bị tách quá nhiều segment nhỏ**: tăng `--hash-threshold 6`
