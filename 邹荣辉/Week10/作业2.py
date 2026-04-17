import os
import base64
from openai import OpenAI
from pdf2image import convert_from_path

#1. 配置 API
API_KEY = "sk-d49ac6c9a8c440a695c83dd86053c33b"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

#2. PDF 转图片
def pdf_first_page_to_base64(pdf_path: str) -> str:
    """
    将 PDF 文件的第一页转换为 Base64 编码的图片数据（PNG 格式）。
    """
    # 将第一页转换为 PIL 图像（分辨率可自行调节）
    images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=200)
    if not images:
        raise RuntimeError("无法读取 PDF 第一页")

    # 将 PIL 图像转为 Base64
    from io import BytesIO
    img_buffer = BytesIO()
    images[0].save(img_buffer, format="PNG")
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


pdf_file = "C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/Week10/邹荣辉_大模型应用开发实习生_简历.pdf"
image_base64 = pdf_first_page_to_base64(pdf_file)

# 3. 调用 Qwen-VL 模型
def analyze_pdf_page(image_base64: str, prompt: str = "请描述这张图片的内容") -> str:
    """
    调用 Qwen-VL 模型对图片进行分析。
    """
    completion = client.chat.completions.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_base64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        stream=False,
    )
    return completion.choices[0].message.content

# 自定义提示词，可要求模型以特定格式输出
prompt_text = """
请提取这张 PDF 页面中的所有文字信息，并整理为结构化的 JSON 格式。
如果包含表格，请一并转换成 Markdown 表格格式。
"""

result = analyze_pdf_page(image_base64, prompt=prompt_text)
print("模型返回结果：")
print(result)