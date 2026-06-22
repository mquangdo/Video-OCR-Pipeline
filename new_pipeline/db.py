"""
Database layer cho Video OCR Pipeline.
Dùng psycopg2 thuần (không ORM) để giữ đơn giản, dễ debug.

Cài đặt:
    pip install psycopg2-binary

Khởi tạo schema:
    python -c "import db; db.init_schema()"
    (hoặc tự chạy schema.sql bằng psql)

Biến môi trường:
    DATABASE_URL = postgresql://user:password@host:5432/dbname
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


@contextmanager
def get_conn():
    """Context manager: tự commit nếu thành công, rollback nếu lỗi."""
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
    """Tạo bảng nếu chưa có. Gọi 1 lần khi setup, hoặc tự động mỗi lần chạy
    pipeline (CREATE TABLE IF NOT EXISTS nên an toàn để gọi lại nhiều lần)."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy schema.sql tại {_SCHEMA_PATH}")

    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
    log.info("Database schema đã sẵn sàng (DSN=%s)", DB_DSN.split("@")[-1])


# ---------------------------------------------------------------------------
# Video record
# ---------------------------------------------------------------------------

def get_video_by_url(youtube_url: str) -> Optional[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM videos WHERE youtube_url = %s", (youtube_url,))
            return cur.fetchone()


def create_or_get_video(youtube_url: str, title: str = None) -> dict:
    """Idempotency entry point.
    Nếu URL đã có trong DB -> trả về record cũ (kèm status hiện tại).
    Nếu chưa có -> tạo record mới với status='pending'.
    """
    existing = get_video_by_url(youtube_url)
    if existing:
        return existing

    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO videos (youtube_url, title, status)
                VALUES (%s, %s, 'pending')
                ON CONFLICT (youtube_url) DO UPDATE SET youtube_url = EXCLUDED.youtube_url
                RETURNING *
                """,
                (youtube_url, title),
            )
            return cur.fetchone()


def update_video_status(video_id: int, status: str, error_message: str = None, **fields):
    """Update status + các field tuỳ chọn khác (video_path, video_fps, duration...)."""
    set_clauses = ["status = %s", "updated_at = now()"]
    values = [status]

    if error_message is not None:
        set_clauses.append("error_message = %s")
        values.append(error_message)

    for key, value in fields.items():
        set_clauses.append(f"{key} = %s")
        values.append(value)

    values.append(video_id)
    sql = f"UPDATE videos SET {', '.join(set_clauses)} WHERE id = %s"

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, values)


def reset_video_for_reprocess(video_id: int):
    """Đặt lại status='pending' và xoá lỗi cũ — dùng khi chạy lại với --force."""
    update_video_status(video_id, "pending", error_message=None)


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------

def insert_segments(video_id: int, segments: list[dict]):
    """Bulk insert segments. `segments` là list dict đúng format của segments_to_json."""
    if not segments:
        return

    rows = [
        (
            video_id,
            seg["start"],
            seg["end"],
            seg.get("duration"),
            seg.get("start_frame"),
            seg.get("end_frame"),
            seg.get("text"),
            seg.get("audio_file"),
        )
        for seg in segments
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO segments
                    (video_id, start_time, end_time, duration, start_frame, end_frame, text, audio_file)
                VALUES %s
                """,
                rows,
            )
    log.info("Đã insert %d segments vào DB cho video_id=%s", len(rows), video_id)


def delete_segments_for_video(video_id: int):
    """Xoá segments cũ trước khi insert lại — dùng khi reprocess."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM segments WHERE video_id = %s", (video_id,))


def search_segments_by_text(query: str, limit: int = 20) -> list[dict]:
    """Full-text search nội dung subtitle. Ví dụ: search_segments_by_text("nông dân")."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT s.*, v.youtube_url, v.title
                FROM segments s
                JOIN videos v ON v.id = s.video_id
                WHERE to_tsvector('simple', coalesce(s.text, ''))
                      @@ to_tsquery('simple', %s)
                ORDER BY s.created_at DESC
                LIMIT %s
                """,
                (query.replace(" ", " & "), limit),
            )
            return cur.fetchall()