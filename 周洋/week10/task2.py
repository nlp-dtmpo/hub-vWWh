import base64
from openai import OpenAI
from pdf2image import convert_from_path
import io

# 配置
API_KEY = "your-dashscope-api-key"  # 替换为你的 API Key
PDF_PATH = "算法刷题LeetCode中文版.pdf"      # 替换为你的 PDF 路径

# 1. 将 PDF 第一页转换为 base64
images = convert_from_path(PDF_PATH, first_page=1, last_page=1, dpi=200)
buffer = io.BytesIO()
images[0].save(buffer, format='JPEG', quality=95)
base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

# 2. 调用 Qwen-VL
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

response = client.chat.completions.create(
    model='qwen-vl-max',
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "将图片内容提取为 Markdown 格式，输出关键信息。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                },
            ],
        }
    ],
)

# 3. 输出结果
print(response.choices[0].message.content)

# 保存到文件
with open("output.md", "w", encoding="utf-8") as f:
    f.write(response.choices[0].message.content)