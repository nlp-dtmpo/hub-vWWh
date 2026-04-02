import base64
import fitz  # PyMuPDF
from openai import OpenAI


def pdf_first_page_to_png_bytes(pdf_path: str) -> bytes:
    """Render the first page of a PDF to PNG bytes."""
    doc = fitz.open(pdf_path)
    try:
        if len(doc) == 0:
            raise ValueError("PDF has no pages.")
        page = doc[0]
        pix = page.get_pixmap(dpi=200)
        return pix.tobytes("png")
    finally:
        doc.close()


def extract_pdf_first_page_with_qwenvl(
    pdf_path: str,
    api_key: str = "xxxx",
    model: str = "qwen-vl-max-latest",
    prompt: str = "请解析这页PDF内容，并输出结构化摘要：标题、核心要点、关键数据（如有）。",
) -> str:
    """Parse only the first page of a PDF via Qwen-VL API."""
    if not pdf_path:
        raise ValueError("Please set a valid PDF file path in `pdf_path` first.")

    image_bytes = pdf_first_page_to_png_bytes(pdf_path)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    image_data_url = f"data:image/png;base64,{image_b64}"

    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    pdf_path = "./test.pdf"

    result = extract_pdf_first_page_with_qwenvl(pdf_path=pdf_path, api_key="你的api-key")
    print(result)
