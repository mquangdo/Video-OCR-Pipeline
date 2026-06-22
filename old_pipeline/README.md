# Old Pipeline — Video Subtitle OCR (Binary Segmentation)

Phiên bản đầu tiên: dùng thuật toán **binary segmentation** đệ quy để tìm điểm chuyển subtitle, OCR bằng Qwen 2.5 VL qua LangChain + vLLM. Cung cấp CLI batch và REST API.

## Luồng hoạt động

```
Video file (.mp4)
        │
        ▼
[decord] Đọc frame random-access
        │
        ▼
[Crop region] Cắt vùng phụ đề (pixel tuyệt đối: x1=100, y1=530, x2=1200, y2=680)
        │
        ▼
[Scan] OCR mỗi scan_step giây (mặc định 4s)
        │  ┌─ Nếu text thay đổi so với frame trước
        │  │
        │  ▼
        │  [Binary Segmentation] Đệ quy chia đôi khoảng thời gian
        │  cho đến khi ≤ 0.5s → tìm chính xác thời điểm chuyển
        │
        │  ┌─ Nếu text giống (similarity ≥ 0.9)
        │  │
        │  └─ Bỏ qua, tiếp tục scan
        │
        ▼
[Build Segments] Gán start/end cho từng đoạn subtitle
        │
        ▼
[Serialize] Ghi file JSON
```

### Binary Segmentation giải thích

1. Scan video: OCR mỗi 4 giây
2. Khi phát hiện text tại `t=4s` khác text tại `t=0s`:
   - OCR tại `t=2s`
   - Nếu `t=0s` giống `t=2s` → điểm chuyển nằm giữa 2s-4s → đệ quy tiếp
   - Nếu `t=2s` giống `t=4s` → điểm chuyển nằm giữa 0s-2s → đệ quy tiếp
   - Lặp cho đến khi khoảng cách ≤ 0.5s

## Cách chạy

### 1. Cài đặt

```bash
pip install -r requirements.txt
```

Cần thêm hệ thống có cài `ffmpeg` (cho decord/pydub nếu mở rộng).

### 2. Cấu hình vLLM

```bash
# Biến môi trường bắt buộc
export OPENAI_API_BASE="http://<IP_VLLM_SERVER>:8233/v1"
# Mặc định: http://localhost:8233/v1
```

### 3. Chạy CLI batch

```bash
# Chỉnh đường dẫn trong video_processor.py (__main__)
# Hoặc set env var:
export VIDEO_INPUT_DIR="/path/to/videos"
export PROCESSING_OUTPUT_DIR="/path/to/output"

python video_processor.py
```

**Tham số** (sửa trực tiếp trong code `__main__`):

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `BATCH_SIZE` | 4 | Số video xử lý song song |
| `SCAN_STEP` | 4 | Bước nhảy scan (giây) |
| `START_INDEX` | None | Video bắt đầu xử lý |
| `END_INDEX` | None | Video kết thúc xử lý |

### 4. Chạy API Server

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Path | Mô tả |
|--------|------|-------|
| `GET` | `/health` | Health check |
| `POST` | `/process` | Upload video → trả về segments |

**Ví dụ gọi API:**

```bash
# Xử lý video với tham số mặc định
curl -X POST http://localhost:8000/process \
  -F "file=@video.mp4"

# Tùy chỉnh scan_step và crop region
curl -X POST http://localhost:8000/process \
  -F "file=@video.mp4" \
  -F "scan_step=2" \
  -F "crop=100,530,1200,680"
```

**Response:**

```json
{
  "filename": "video.mp4",
  "segment_count": 15,
  "segments": [
    {"id": 0, "start": 0, "end": 4, "text": "Nội dung phụ đề đoạn 1"},
    {"id": 1, "start": 4, "end": 8, "text": "Nội dung phụ đề đoạn 2"}
  ]
}
```

### 5. Chạy bằng Docker

```bash
docker build -t video-ocr-old .

# CLI batch
docker run --rm \
  -e OPENAI_API_BASE=http://host.docker.internal:8233/v1 \
  -v /path/to/videos:/data/raw \
  -v /path/to/output:/data/output \
  video-ocr-old

# API server (ghi đè CMD)
docker run --rm -p 8000:8000 \
  -e OPENAI_API_BASE=http://host.docker.internal:8233/v1 \
  video-ocr-old \
  uvicorn api_server:app --host 0.0.0.0 --port 8000
```

## So sánh sim_threshold

- `binary_segmentation` dùng `sim_threshold=0.9` (90% giống nhau mới merge)
- `is_similar_text` mặc định `sim_threshold=0.85`
- Tăng threshold → ít chia nhỏ hơn (aggressive merge)
- Giảm threshold → nhạy hơn với thay đổi text nhỏ

## Lưu ý

- Crop region hardcoded `(100, 530, 1200, 680)` — chỉ đúng video 1280x720. Video khác kích thước cần điều chỉnh.
- `SSL verify=False` cho httpx client — chỉ an toàn khi gọi vLLM local.
- Không có database persistence — kết quả chỉ lưu file JSON.
- Không idempotent — chạy lại sẽ xử lý lại toàn bộ.
