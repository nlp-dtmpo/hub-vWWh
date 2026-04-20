import base64
import fitz  # PyMuPDF
from PIL import Image
import io
import dashscope
from http import HTTPStatus
import os

client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# PDF 文件路径
pdf_path = "week10.pdf"  # 替换为你的 PDF 路径


def pdf_first_page_to_image(pdf_path, dpi=200):

    # 打开 PDF
    pdf_document = fitz.open(pdf_path)

    # 获取第一页
    page = pdf_document[0]

    # 设置缩放矩阵（DPI / 72 = 缩放因子）
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    # 渲染为 Pixmap
    pix = page.get_pixmap(matrix=matrix)

    # 转换为 PIL Image
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))

    # 转换为 base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    pdf_document.close()

    return img_base64, img.size


# ========== Qwen-VL 调用函数 ==========
def analyze_image_with_qwen_vl(image_base64, prompt="请描述这张图片中的内容"):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "你是一个专业的文档解析助手。"}]
        },
        {
            "role": "user",
            "content": [
                {"image": f"data:image/png;base64,{image_base64}"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    try:
        response = dashscope.MultiModalConversation.call(
            model="qwen-vl-plus",  # 可选: qwen-vl-max, qwen2.5-vl-72b-instruct
            messages=messages,
            api_key=dashscope.api_key
        )

        if response.status_code == HTTPStatus.OK:
            return response.output["choices"][0]["message"]["content"][0]["text"]
        else:
            print(f"API 调用失败: {response.status_code} - {response.message}")
            return None

    except Exception as e:
        print(f"请求异常: {e}")
        return None


# ========== 主程序 ==========
def main():
    # 1. 检查 PDF 文件是否存在
    if not os.path.exists(pdf_path):
        print(f"PDF 文件不存在: {pdf_path}")
        return

    # 2. 将 PDF 第一页转换为图片
    print("正在转换 PDF 第一页为图片...")
    try:
        img_base64, img_size = pdf_first_page_to_image(pdf_path, dpi=150)
        print(f"转换成功，图片尺寸: {img_size}")
    except Exception as e:
        print(f"PDF 转换失败: {e}")
        return

    # 3. 定义分析提示词
    prompt = """
    请仔细分析这张PDF页面，并告诉我：
    1. 这是什么类型的文档？（如：合同、报告、发票、简历等）
    2. 文档的主要内容是什么？
    3. 提取文档中的关键信息（如标题、日期、金额、重要条款等）
    4. 文档中是否有表格？如果有，请描述表格内容
    """

    # 4. 调用 Qwen-VL 分析
    print("正在调用 Qwen-VL 模型分析...")
    result = analyze_image_with_qwen_vl(img_base64, prompt)

    # 5. 输出结果
    if result:
        print("\n" + "=" * 50)
        print("【Qwen-VL 分析结果】")
        print("=" * 50)
        print(result)
        print("=" * 50)
    else:
        print("分析失败，请检查 API Key 和网络连接")


if __name__ == "__main__":
    main()