"""
crawl_db.py — Database layer cho Crawl Pipeline.

Bảng: crawl_status
  - Mỗi row = 1 video YouTube
  - Dùng để check idempotency trước khi xử lý
  - Lưu metadata sau khi OCR xong

Cài đặt:
    pip install psycopg2-binary

Biến môi trường:
    DATABASE_URL = postgresql://postgres:postgres@localhost:5432/video_ocr

Khởi tạo schema:
    python -c "import crawl_db; crawl_db.init_schema()"
"""

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

DB_DSN = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/video_ocr",
)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def get_conn():
    """Context manager: auto commit nếu thành công, rollback nếu lỗi."""
    conn = psycopg2.connect(DB_DSN)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema():
    """Tạo bảng nếu chưa có. An toàn khi gọi lại nhiều lần (IF NOT EXISTS)."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy schema.sql tại {_SCHEMA_PATH}")
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    log.info("Schema crawl_status sẵn sàng")


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------

def get_video(video_id: str) -> Optional[dict]:
    """Lấy row theo video_id. Trả về None nếu chưa có."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM crawl_status WHERE video_id = %s",
                (video_id,),
            )
            return cur.fetchone()


def is_done(video_id: str) -> bool:
    """Trả về True nếu video đã xử lý xong (status = 'done')."""
    row = get_video(video_id)
    return row is not None and row["status"] == "done"


def create_or_get(
    video_id: str,
    url: str,
    playlist_url: Optional[str] = None,
    playlist_index: Optional[int] = None,
) -> dict:
    """
    Idempotency entry point:
    - Nếu chưa có → INSERT row mới với status='pending'
    - Nếu đã có    → trả về row cũ (kèm status hiện tại)
    """
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO crawl_status (video_id, url, playlist_url, playlist_index, status)
                VALUES (%s, %s, %s, %s, 'pending')
                ON CONFLICT (video_id) DO UPDATE
                    SET video_id = EXCLUDED.video_id   -- no-op, chỉ để RETURNING hoạt động
                RETURNING *
                """,
                (video_id, url, playlist_url, playlist_index),
            )
            return cur.fetchone()


# ---------------------------------------------------------------------------
# Status updates
# ---------------------------------------------------------------------------

def mark_processing(video_id: str):
    """Đánh dấu bắt đầu xử lý."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE crawl_status
                SET status = 'processing', started_at = now(), error_message = NULL
                WHERE video_id = %s
                """,
                (video_id,),
            )


def mark_done(
    video_id: str,
    output_dir: str,
    segment_count: int,
    total_duration: float,
):
    """Lưu metadata sau khi OCR xong thành công."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE crawl_status
                SET status         = 'done',
                    finished_at    = now(),
                    output_dir     = %s,
                    segment_count  = %s,
                    total_duration = %s,
                    error_message  = NULL
                WHERE video_id = %s
                """,
                (output_dir, segment_count, total_duration, video_id),
            )


def mark_failed(video_id: str, error_message: str):
    """Lưu lỗi khi pipeline thất bại."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE crawl_status
                SET status        = 'failed',
                    finished_at   = now(),
                    error_message = %s,
                    retry_count   = retry_count + 1
                WHERE video_id = %s
                """,
                (error_message, video_id),
            )


def reset(video_id: str):
    """Reset về pending để chạy lại (dùng khi --force)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE crawl_status
                SET status        = 'pending',
                    started_at    = NULL,
                    finished_at   = NULL,
                    error_message = NULL,
                    output_dir    = NULL,
                    segment_count = NULL,
                    total_duration = NULL
                WHERE video_id = %s
                """,
                (video_id,),
            )


# ---------------------------------------------------------------------------
# Queries tiện ích
# ---------------------------------------------------------------------------

def get_all_by_status(status: str) -> list[dict]:
    """Lấy tất cả video theo status. Ví dụ: get_all_by_status('failed')."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM crawl_status WHERE status = %s ORDER BY created_at",
                (status,),
            )
            return cur.fetchall()


def summary() -> dict:
    """Thống kê nhanh số video theo từng status."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*) as count
                FROM crawl_status
                GROUP BY status
                ORDER BY status
                """
            )
            return {row[0]: row[1] for row in cur.fetchall()}