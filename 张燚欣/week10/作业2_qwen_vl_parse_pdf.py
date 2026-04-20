import os
import base64
import io
from openai import OpenAI

# 尝试导入PDF处理库
try:
    import pypdfium2 as pdfium

    PDF_SUPPORT = True
    print("✓ PDF支持库已加载")
except ImportError:
    PDF_SUPPORT = False
    print("✗ 未安装pypdfium2，请运行: pip install pypdfium2")
    exit(1)

print("=" * 60)
print("Qwen-VL PDF解析程序")
print("=" * 60)

# ==================== 配置部分 ====================

API_KEY = "sk-c9b1982f0e674957ba9da72ce95922d6"  # ← 请在这里填入您的真实API Key！
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-vl-plus"  # 或使用 qwen3-vl-flash

# 您的PDF文件路径
PDF_PATH = r"D:\badou\第10周：多模态大模型\应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"

# 您想问模型的问题
QUESTION = "请详细解析这张PDF页面中的内容，总结主要观点和关键信息。"

print(f"\n配置信息:")
print(f"  API Key: {API_KEY[:10]}..." if API_KEY != "sk-xxxxxx" else "  API Key: 未设置！")
print(f"  模型: {MODEL_NAME}")
print(f"  PDF文件: {os.path.basename(PDF_PATH)}")
print(f"  完整路径: {PDF_PATH}")

# 检查API Key
if API_KEY == "sk-xxxxxx":
    print("\n❌ 错误: 请先设置正确的API Key！")
    print("   请在代码中找到 'API_KEY = \"sk-xxxxxx\"' 这一行")
    print("   将 'sk-xxxxxx' 替换为您的真实API Key")
    print("\n   获取API Key: https://help.aliyun.com/zh/model-studio/get-api-key")
    input("\n按回车键退出...")
    exit(1)


# ==================== 将PDF第一页转换为Base64图片 ====================
def pdf_first_page_to_base64(pdf_path, scale=2):
    """将PDF第一页转换为Base64图片"""
    print(f"\n[步骤1] 正在处理PDF文件...")

    # 检查文件是否存在
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")

    # 获取文件大小
    file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
    print(f"  ✓ PDF文件存在，大小: {file_size:.2f} MB")

    # 打开PDF
    print("  - 正在打开PDF文件...")
    pdf = pdfium.PdfDocument(pdf_path)

    # 获取总页数
    num_pages = len(pdf)
    print(f"  - PDF总页数: {num_pages}")

    # 获取第一页
    print("  - 正在读取第一页...")
    page = pdf[0]

    # 渲染为位图
    print("  - 正在渲染图片（这可能需要几秒钟）...")
    bitmap = page.render(scale=scale)

    # 转换为PIL图像
    print("  - 正在转换图像格式...")
    pil_image = bitmap.to_pil()

    # 获取图片尺寸
    width, height = pil_image.size
    print(f"  - 图片尺寸: {width} x {height} 像素")

    # 转换为Base64
    print("  - 正在编码为Base64...")
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=85)
    img_base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # 计算Base64大小
    img_size = len(img_base64_str) / (1024 * 1024)  # MB
    print(f"  - Base64编码大小: {img_size:.2f} MB")

    # 关闭PDF
    pdf.close()

    print("  ✓ PDF第一页转换完成！")
    return f"data:image/jpeg;base64,{img_base64_str}"


# ==================== 调用Qwen-VL模型 ====================
def parse_image_with_qwen_vl(image_data_url, question):
    """使用Qwen-VL模型解析图片"""
    print(f"\n[步骤2] 正在调用Qwen-VL模型...")
    print(f"  模型: {MODEL_NAME}")
    print(f"  问题: {question}")

    # 初始化OpenAI客户端
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )

    try:
        print("  - 正在发送请求到阿里云（这可能需要10-30秒）...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_data_url},
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ],
            stream=False,
            max_tokens=2000,  # 限制输出长度
        )

        # 获取结果
        result = response.choices[0].message.content

        # 获取token使用情况
        usage = response.usage
        print(f"\n  ✓ API调用成功！")
        print(f"  输入Token数: {usage.prompt_tokens}")
        print(f"  输出Token数: {usage.completion_tokens}")
        print(f"  总计Token数: {usage.total_tokens}")

        print("\n" + "=" * 60)
        print("模型解析结果")
        print("=" * 60)
        print(result)
        print("=" * 60)
        return result

    except Exception as e:
        print(f"\n❌ API调用失败: {e}")
        print("\n可能的原因:")
        print("1. API Key不正确或已过期")
        print("2. 账户余额不足")
        print("3. 网络连接问题")
        print("4. 图片太大超出限制")

        # 提供更多调试信息
        print("\n调试信息:")
        print(f"  - API Key前缀: {API_KEY[:15]}...")
        print(f"  - Base URL: {BASE_URL}")
        print(f"  - 模型: {MODEL_NAME}")
        return None


# ==================== 主程序 ====================
if __name__ == "__main__":
    try:
        # 检查PDF文件
        if not os.path.exists(PDF_PATH):
            print(f"\n❌ 错误: PDF文件不存在!")
            print(f"   路径: {PDF_PATH}")
            print("\n请检查:")
            print("1. 文件路径是否正确")
            print("2. 文件名是否包含中文（已支持，但请确认）")
            print("3. 文件是否确实在该位置")
            input("\n按回车键退出...")
            exit(1)

        # 转换PDF第一页为图片
        img_url = pdf_first_page_to_base64(PDF_PATH, scale=1.5)  # scale=1.5 平衡清晰度和大小

        # 调用模型解析
        result = parse_image_with_qwen_vl(img_url, QUESTION)

        if result:
            print("\n✓ 程序执行成功！")
        else:
            print("\n✗ 程序执行失败，请检查上方错误信息")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    input("按回车键退出...")

    # 在代码最后添加保存功能
    with open("解析结果.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("结果已保存到 解析结果.txt")