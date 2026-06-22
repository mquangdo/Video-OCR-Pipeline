"""
Module kết nối MinIO để lưu audio clip thay cho local disk.

Cài đặt:
    pip install minio

Biến môi trường:
    MINIO_ENDPOINT     = host:port, ví dụ "localhost:9000"
    MINIO_ACCESS_KEY   = minioadmin (mặc định, đổi nếu đã set khác)
    MINIO_SECRET_KEY   = minioadmin123
    MINIO_BUCKET        = video-ocr-audio (tự tạo nếu chưa có)
    MINIO_SECURE        = "false" nếu chạy http (mặc định), "true" nếu https
"""

import os
import logging
from datetime import timedelta
from pathlib import Path

from minio import Minio
from minio.error import S3Error

log = logging.getLogger(__name__)

MINIO_ENDPOINT   = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin123")
MINIO_BUCKET     = os.getenv("MINIO_BUCKET", "video-ocr-audio")
MINIO_SECURE     = os.getenv("MINIO_SECURE", "false").lower() == "true"

_client: Minio = None


def get_client() -> Minio:
    global _client
    if _client is None:
        _client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=MINIO_SECURE,
        )
    return _client


def ensure_bucket(bucket: str = MINIO_BUCKET):
    """Tạo bucket nếu chưa tồn tại. Gọi an toàn nhiều lần."""
    client = get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log.info("Đã tạo bucket MinIO: %s", bucket)


def upload_file(local_path: Path, object_name: str, bucket: str = MINIO_BUCKET) -> str:
    """Upload 1 file lên MinIO. Trả về object_name (dùng để lưu vào DB)."""
    client = get_client()
    client.fput_object(bucket, object_name, str(local_path))
    log.info("Uploaded %s -> minio://%s/%s", local_path.name, bucket, object_name)
    return object_name


def download_file(object_name: str, dest_path: Path, bucket: str = MINIO_BUCKET):
    """Tải 1 file từ MinIO về local."""
    client = get_client()
    client.fget_object(bucket, object_name, str(dest_path))


def get_presigned_url(object_name: str, expires_seconds: int = 3600,
                       bucket: str = MINIO_BUCKET) -> str:
    """Tạo URL tạm để nghe/tải file trực tiếp qua browser (mặc định hết hạn sau 1h)."""
    client = get_client()
    return client.presigned_get_object(
        bucket, object_name, expires=timedelta(seconds=expires_seconds)
    )


def delete_file(object_name: str, bucket: str = MINIO_BUCKET):
    client = get_client()
    client.remove_object(bucket, object_name)