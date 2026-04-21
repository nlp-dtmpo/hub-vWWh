import base64

import fitz
from PIL import Image
from openai import OpenAI
import os


def pdf_first_page_to_image(pdf_path: str, output_image_path: str = None) -> Image:
    """
    将 PDF 第一页转换为 PIL Image 对象

    Args:
        pdf_path: PDF 文件路径
        output_image_path: 可选，保存图片的路径

    Returns:
        PIL Image 对象
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

    # 打开 PDF 文件
    doc = fitz.open(pdf_path)

    # 加载第一页（页码从 0 开始）
    page = doc.load_page(0)

    # 获取页面图像，设置缩放系数（zoom=2 即 144 DPI，平衡清晰度和文件大小）
    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # 将图像数据转为 PIL Image
    generate_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 可选：保存到本地
    if output_image_path:
        generate_img.save(output_image_path)

    doc.close()
    return generate_img


def image_to_base64_url(image: Image, format: str = "PNG") -> str:
    """
    将 PIL Image 转换为 base64 格式的 data URL

    Args:
        image: PIL Image 对象
        format: 输出格式，PNG 或 JPEG

    Returns:
        data:image/...;base64,xxx 格式的字符串
    """
    import io

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    mime_type = "image/png" if format.lower() == "png" else "image/jpeg"
    return f"data:{mime_type};base64,{img_base64}"


# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key="sk-**",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""  # 定义完整回复
is_answering = False  # 判断是否结束思考过程并开始回复

FDF_PATH = "./Week10-多模态大模型.pdf"
print(f"正在读取 PDF: {FDF_PATH}")

img = pdf_first_page_to_image(FDF_PATH)

# 2. 转为 base64 data URL
img_base64_url = image_to_base64_url(img, format="PNG")

# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-flash",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64_url
                    },
                },
                {"type": "text", "text": "请详细描述这张图片的内容。"},
            ],
        },
    ],
    stream=True,
    # 解除以下注释会在最后一个chunk返回Token使用量
    stream_options={
        "include_usage": True
    }
)

print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
print(reasoning_content)
print("=" * 20 + "完整回复" + "=" * 20 + "\n")
print(answer_content)
