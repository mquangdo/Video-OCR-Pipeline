-- ============================================================
-- Crawl Pipeline Schema
-- ============================================================

CREATE TABLE IF NOT EXISTS crawl_status (
    id              SERIAL PRIMARY KEY,

    -- Identity
    video_id        TEXT NOT NULL UNIQUE,   -- YouTube video ID, e.g. "dQw4w9WgXcQ"
    url             TEXT NOT NULL,

    -- Playlist context
    playlist_url    TEXT,
    playlist_index  INTEGER,                -- index trong playlist (0-based)

    -- Processing state
    status          TEXT NOT NULL DEFAULT 'pending',
    -- pending | processing | done | failed

    -- Output location
    output_dir      TEXT,                   -- path tới folder crawled_data/<video_id>/
    segment_count   INTEGER,                -- số segment sau khi OCR xong
    total_duration  FLOAT,                  -- tổng duration (giây) của toàn bộ segment

    -- Error tracking
    error_message   TEXT,
    retry_count     INTEGER NOT NULL DEFAULT 0,

    -- Timestamps
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at      TIMESTAMPTZ,            -- lúc bắt đầu xử lý
    finished_at     TIMESTAMPTZ             -- lúc done hoặc failed
);

-- Index thường dùng
CREATE INDEX IF NOT EXISTS idx_crawl_status_status   ON crawl_status (status);
CREATE INDEX IF NOT EXISTS idx_crawl_status_playlist ON crawl_status (playlist_url);