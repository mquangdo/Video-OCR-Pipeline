-- ============================================================================
-- Schema: Video OCR Pipeline
-- ============================================================================

CREATE TABLE IF NOT EXISTS videos (
    id              SERIAL PRIMARY KEY,
    youtube_url     TEXT UNIQUE NOT NULL,
    title           TEXT,
    video_path      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    -- status: pending -> downloading -> extracting -> ocr_processing
    --         -> audio_extracting -> done | failed
    error_message   TEXT,
    video_fps       REAL,
    duration        REAL,
    created_at      TIMESTAMP NOT NULL DEFAULT now(),
    updated_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS segments (
    id              SERIAL PRIMARY KEY,
    video_id        INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    start_time      REAL NOT NULL,
    end_time        REAL NOT NULL,
    duration        REAL,
    start_frame     INTEGER,
    end_frame       INTEGER,
    text            TEXT,
    audio_file      TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_segments_video_id ON segments(video_id);

-- Full-text search trên nội dung subtitle (tiếng Việt nên dùng config 'simple'
-- vì Postgres không có dictionary tiếng Việt built-in; nếu cần stemming tiếng
-- Việt tốt hơn, xem xét extension pg_jieba/unaccent kết hợp).
CREATE INDEX IF NOT EXISTS idx_segments_text_fts
    ON segments USING GIN (to_tsvector('simple', coalesce(text, '')));

CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);