# -*- coding: utf-8 -*-
"""
功能：
1. 读取本地 PDF
2. 提取第一页并转成图片
3. 调用 Qwen-VL 解析
4. 输出结构化结果

依赖：
pip install pymupdf pillow openai
"""

import fitz  # PyMuPDF
import base64
from openai import OpenAI
import os


# =========================
# 1. PDF → 图片（第一页）
# =========================
def pdf_to_image(pdf_path, output_path="page1.png"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        raise ValueError("PDF没有任何页面")

    page = doc.load_page(0)  # 第一页
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 提高分辨率

    pix.save(output_path)
    doc.close()

    return output_path


# =========================
# 2. 图片 → Base64
# =========================
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =========================
# 3. 调用 Qwen-VL
# =========================
def call_qwen_vl(image_path, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    )

    base64_image = encode_image(image_path)

    response = client.chat.completions.create(
        model="qwen3-vl-plus",  # 推荐模型
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的文档解析助手"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    },
                    {
                        "type": "text",
                        "text": (
                            "请解析该PDF页面内容，并按JSON格式输出：\n"
                            "{\n"
                            "  \"title\": \"标题\",\n"
                            "  \"summary\": \"摘要\",\n"
                            "  \"keywords\": [\"关键词1\", \"关键词2\"],\n"
                            "  \"content\": \"主要内容\"\n"
                            "}"
                        )
                    }
                ]
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def main():

    pdf_path = "test.pdf"
    image_path = "page1.png"
    api_key = "xxxxxxx"

    try:
        print("PDF 转图片...")
        img = pdf_to_image(pdf_path, image_path)
        print(f"生成图片: {img}")

        print("调用 Qwen-VL 解析...")
        result = call_qwen_vl(img, api_key)

        print("解析结果：")
        print(result)

    except Exception as e:
        print("出错：", str(e))


if __name__ == "__main__":
    main()