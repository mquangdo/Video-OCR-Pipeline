# Video Subtitle OCR Pipeline

Hệ thống trích xuất phụ đề burned-in từ video bằng Qwen 2.5 VL (Vision-Language) chạy trên vLLM.

## Kiến trúc tổng thể

```
Video-OCR-Pipeline/
├── old_pipeline/        ← Phiên bản 1: Binary segmentation + FastAPI
├── new_pipeline/        ← Phiên bản 2: Perceptual hash + PostgreSQL + MinIO
└── README.md            ← Bạn đang ở đây
```

| | old_pipeline | new_pipeline |
|---|---|---|
| Phát hiện subtitle | Binary segmentation đệ quy (OCR nhiều lần) | Perceptual hash (1 OCR/group) |
| Số OCR calls | Nhiều | Ít (tiết kiệm hơn) |
| So sánh text | Fuzzy (SequenceMatcher) | Exact string match |
| API Server | FastAPI (upload file) | CLI only |
| Batch processing | ThreadPoolExecutor | 1 video/lần |
| Output | File JSON | JSON + PostgreSQL + MinIO |
| Idempotency | Không | Có (DB status tracking) |
| Audio extraction | Không | pydub + MinIO |

## Luồng hoạt động (sơ đồ chung)

```
Video YouTube / File video
        │
        ▼
  [Download / Đọc video]
        │
        ▼
  [Trích frame theo FPS]
        │
        ▼
  [Crop vùng phụ đề]
        │
        ▼
  [Nhóm frame cùng subtitle]
        │
        ▼
  [OCR bằng Qwen 2.5 VL (vLLM)]
        │
        ▼
  [Build segments: start, end, text]
        │
        ├──→ output.json
        ├──→ (new_pipeline) PostgreSQL
        └──→ (new_pipeline) MinIO audio clips
```

## Chạy nhanh

### Yêu cầu chung

- Python 3.11+
- vLLM server đang chạy với model `Qwen/Qwen2.5-VL-3B-Instruct`

### Chạy old_pipeline (xem chi tiết trong `old_pipeline/README.md`)

```bash
cd old_pipeline
pip install -r requirements.txt

# CLI batch
python video_processor.py

# Hoặc chạy API server
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Chạy new_pipeline (xem chi tiết trong `new_pipeline/README.md`)

```bash
cd new_pipeline
pip install -r requirements.txt

# Khởi tạo database
docker compose up -d postgres minio    # chạy PostgreSQL + MinIO
python -c "import db; db.init_schema()"

# Chạy pipeline
python video_ocr.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

## Chi tiết từng phiên bản

- [`old_pipeline/README.md`](old_pipeline/README.md) — Binary segmentation + FastAPI
- [`new_pipeline/README.md`](new_pipeline/README.md) — Perceptual hash + PostgreSQL + MinIO
