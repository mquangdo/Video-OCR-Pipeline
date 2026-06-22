import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host="localhost",      # hoặc "postgres" nếu chạy cùng network Docker
    port=5432,
    dbname="video_ocr",
    user="postgres",
    password="postgres",
)

df_videos = pd.read_sql("SELECT * FROM videos", conn)
print(df_videos)