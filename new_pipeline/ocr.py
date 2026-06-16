import base64
import requests
from pathlib import Path

def ocr_image(image_path: str, server_url: str = "http://localhost:8000") -> str:
    # Đọc và encode ảnh sang base64 để phù hợp với định dạng JSON trong giao th
    image_data = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    
    # Xác định media type
    suffix = Path(image_path).suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg", 
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")

    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2.5-VL-3B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Hãy đọc và trích xuất toàn bộ văn bản trong ảnh này. Chỉ trả về nội dung văn bản, không giải thích thêm."
                    }
                ]
            }],
            "max_tokens": 1024,
            "temperature": 0.1,
        }
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    result = ocr_image("ocr_img.png")
    print(result)