# New Pipeline — Video Subtitle OCR (Perceptual Hash + PostgreSQL + MinIO)

Phiên bản 2: dùng **perceptual hash** để nhóm frame cùng subtitle, chỉ OCR 1 frame đại diện mỗi nhóm — tiết kiệm API calls. Kết quả lưu vào PostgreSQL + MinIO, hỗ trợ idempotency (chạy lại không xử lý video đã done).

## Luồng hoạt động

```
YouTube URL
        │
        ▼
[Step 1] yt-dlp → Download video .mp4
        │
        ▼
[Step 2] OpenCV → Extract frames @ 1fps
        │        Tính timestamp + frame_index
        │
        ▼
[Step 3] PIL → Crop bottom 28% (vùng subtitle)
        │
        ▼
[Step 4] imagehash → Tính pHash cho mỗi cropped frame
        │        Nhóm frame liên tiếp có hash gần giống (Hamming ≤ 4)
        │
        ▼
[Step 5] vLLM (Qwen 2.5 VL) → OCR 1 frame đại diện / nhóm
        │        Gán text cho tất cả frame trong nhóm
        │
        ▼
[Step 6] Build Segments → Gộp nhóm liên tiếp cùng text
        │         Bỏ segment ngắn (< 0.5s) hoặc text rỗng
        │
        ▼
[Step 7] pydub → Extract audio clip cho từng segment
        │        Upload lên MinIO (hoặc fallback local)
        │
        ▼
[Output]
  ├── output.json          ← {start, end, duration, text, audio_file}
  ├── PostgreSQL videos    ← status tracking + metadata
  ├── PostgreSQL segments  ← subtitle data + audio path
  └── MinIO bucket         ← audio .wav files
```

## Cách chạy

### 1. Khởi động hạ tầng (Docker)

```bash
docker compose up -d
```

Sẽ chạy 3 service:

| Service | Port | Vai trò |
|---------|------|---------|
| vLLM | 8000 | Serve model Qwen2.5-VL-3B-Instruct (cần GPU NVIDIA) |
| PostgreSQL | 5432 | Database `video_ocr` (user: postgres / pass: postgres) |
| MinIO | 9000 (API) / 9002 (Console) | Object storage cho audio clips (user: minioadmin / pass: minioadmin123) |

**Lưu ý:** vLLM cần GPU với driver NVIDIA + nvidia-docker runtime.

### 2. Cài đặt Python dependencies

```bash
pip install -r requirements.txt
```

Cần thêm hệ thống có cài `ffmpeg` (pydub yêu cầu để đọc audio từ video).

### 3. Cấu hình (biến môi trường)

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `VLLM_BASE_URL` | `http://192.168.1.100:8000` | URL vLLM server |
| `VLLM_MODEL` | `Qwen/Qwen2.5-VL-3B-Instruct` | Tên model |
| `VLLM_API_KEY` | `token-abc123` | API key vLLM |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/video_ocr` | PostgreSQL connection string |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO host:port |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin123` | MinIO secret key |
| `MINIO_BUCKET` | `video-ocr-audio` | Bucket name (tự tạo nếu chưa có) |
| `MINIO_SECURE` | `false` | Dùng HTTPS hay HTTP |

### 4. Khởi tạo database schema

```bash
# Tự động (lần đầu chạy pipeline)
python video_ocr.py URL --no-db-init=false   # mặc định đã tự init

# Hoặc thủ công
python -c "import db; db.init_schema()"
```

### 5. Chạy pipeline

```bash
# Cơ bản
python video_ocr.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Tùy chỉnh output
python video_ocr.py "URL" -o result.json --audio-dir ./audio

# Tùy chỉnh vLLM server
python video_ocr.py "URL" --vllm-url http://192.168.1.50:8000 --model Qwen/Qwen2.5-VL-3B-Instruct

# Tăng FPS (phụ đề thay đổi nhanh)
python video_ocr.py "URL" --fps 2

# Mở rộng vùng crop (phụ đề cao hơn bình thường)
python video_ocr.py "URL" --subtitle-fraction 0.35

# Xử lý lại video đã done
python video_ocr.py "URL" --force

# Không dùng MinIO, chỉ lưu audio local
python video_ocr.py "URL" --no-minio

# Giữ file tạm để debug
python video_ocr.py "URL" --keep-temp --work-dir ./debug
```

### 6. Tham số CLI đầy đủ

| Flag | Mặc định | Mô tả |
|------|----------|-------|
| `url` | (bắt buộc) | YouTube video URL |
| `-o / --output` | `output.json` | File JSON output |
| `--vllm-url` | env `VLLM_BASE_URL` | URL vLLM server |
| `--model` | env `VLLM_MODEL` | Tên model OCR |
| `--fps` | `1.0` | Số frame trích xuất mỗi giây |
| `--subtitle-fraction` | `0.28` | Tỉ lệ crop phía dưới frame |
| `--hash-threshold` | `4` | Ngưỡng Hamming distance (0 = giống hệt) |
| `--audio-dir` | `<output_dir>/audio/` | Thư mục lưu audio clips |
| `--work-dir` | system temp | Thư mục làm việc tạm |
| `--keep-temp` | False | Giữ file tạm sau xử lý |
| `--force` | False | Xử lý lại video dù đã done |
| `--no-minio` | False | Không upload audio lên MinIO |
| `--no-db-init` | False | Bỏ qua tự tạo schema |

## Output JSON

```json
[
  {
    "start": 1.0,
    "end": 7.0,
    "duration": 6.0,
    "start_frame": 25,
    "end_frame": 175,
    "text": "Nội dung phụ đề đoạn 1",
    "audio_file": "video_1/segment_0000_f25-f175.wav"
  },
  {
    "start": 8.0,
    "end": 12.0,
    "duration": 4.0,
    "start_frame": 200,
    "end_frame": 300,
    "text": "Nội dung phụ đề đoạn 2",
    "audio_file": "video_1/segment_0001_f200-f300.wav"
  }
]
```

- `audio_file`: Nếu dùng MinIO → giá trị là object key (e.g. `video_1/segment_0000_f25-f175.wav`). Nếu MinIO down → fallback lưu đường dẫn file local.

## Database schema

### Bảng `videos`

| Column | Type | Mô tả |
|--------|------|-------|
| `id` | SERIAL PK | Auto-increment ID |
| `youtube_url` | TEXT UNIQUE | URL video |
| `title` | TEXT | Tiêu đề |
| `video_path` | TEXT | Đường dẫn file video |
| `status` | TEXT | Lifecycle: `pending → downloading → extracting → ocr_processing → audio_extracting → done \| failed` |
| `error_message` | TEXT | Lỗi nếu failed |
| `video_fps` | REAL | FPS video |
| `duration` | REAL | Thời lượng (giây) |
| `created_at` | TIMESTAMP | Thời tạo |
| `updated_at` | TIMESTAMP | Thời cập nhật |

### Bảng `segments`

| Column | Type | Mô tả |
|--------|------|-------|
| `id` | SERIAL PK | Auto-increment ID |
| `video_id` | INTEGER FK | Tham chiếu → `videos(id)`, ON DELETE CASCADE |
| `start_time` | REAL | Thời điểm bắt đầu (giây) |
| `end_time` | REAL | Thời điểm kết thúc (giây) |
| `duration` | REAL | Thời lượng |
| `start_frame` | INTEGER | Frame index bắt đầu |
| `end_frame` | INTEGER | Frame index kết thúc |
| `text` | TEXT | Nội dung subtitle OCR |
| `audio_file` | TEXT | MinIO object key hoặc local path |
| `created_at` | TIMESTAMP | Thời tạo |

**Index:** `idx_segments_video_id`, `idx_segments_text_fts` (GIN full-text search), `idx_videos_status`

## Idempotency

1. Chạy lần đầu: tạo record video `status=pending` → xử lý → `status=done`
2. Chạy lại cùng URL: phát hiện `status=done` → bỏ qua
3. Muốn xử lý lại: `--force` → reset `status=pending` → xoá segments cũ → xử lý lại

## Tuning gợi ý

| Tình huống | Giải pháp |
|------------|-----------|
| Phụ đề thay đổi nhanh (< 1s/câu) | Tăng `--fps 2` hoặc `--fps 3` |
| Crop bị thiếu text | Tăng `--subtitle-fraction 0.35` |
| Nhận nhầm 2 phụ đề khác nhau là 1 | Giảm `--hash-threshold 2` |
| Bị tách quá nhiều segment nhỏ | Tăng `--hash-threshold 6` |

## Kiểm tra database

```bash
# Script tiện lợi
python check_db.py

# Hoặc dùng psql
psql -h localhost -U postgres -d video_ocr -c "SELECT * FROM videos;"
psql -h localhost -U postgres -d video_ocr -c "SELECT * FROM segments WHERE video_id = 1;"
```

## MinIO Console

Truy cập `http://localhost:9002` với user `minioadmin` / password `minioadmin123` để xem/dowload audio clips.
