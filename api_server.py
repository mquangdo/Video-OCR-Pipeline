import os
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from video_processor import process_video_to_segments


app = FastAPI(title="Video OCR Processor API", version="1.0.0")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/process")
async def process_video(
    file: UploadFile = File(...),
    scan_step: int = Form(default=4),
    crop: Optional[str] = Form(default=None),  # "x1,y1,x2,y2"
):
    
    # Validate scan_step
    if scan_step < 1:
        raise HTTPException(status_code=400, detail="scan_step must be >= 1")
    
    # Parse crop nếu có
    crop_tuple = None
    if crop:
        try:
            parts = [int(x.strip()) for x in crop.split(",")]
            if len(parts) != 4:
                raise ValueError()
            crop_tuple = tuple(parts)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="crop must be 4 comma-separated integers: x1,y1,x2,y2"
            )
    
    # Lưu file tạm
    suffix = Path(file.filename).suffix if file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
    
    try:
        # Xử lý video
        segments = process_video_to_segments(
            video_path=tmp_path,
            scan_step=scan_step,
            crop=crop_tuple,
        )
        
        return {
            "filename": file.filename,
            "segment_count": len(segments),
            "segments": segments,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Xóa file tạm
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

