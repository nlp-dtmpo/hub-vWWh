from openai import OpenAI
import os
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import base64

API_KEY = "sk-XXXX"  # 阿里云百炼API Key

def pdf_first_page_to_image(pdf_path, zoom=2.0, page_number=1):
    """
    将PDF的第一页转换为图片（base64格式）
    
    Args:
        pdf_path: PDF文件路径
        zoom: 图片缩放比例，默认2.0（提高分辨率）
        page_number: 要转换的页面编号，默认为1（第一页）
    
    Returns:
        base64编码的图片字符串，可以直接用于API调用
    """
    try:
        # 打开PDF文件
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1] # 获取指定页面（注意页码从0开始）
        mat = fitz.Matrix(zoom, zoom) # 设置缩放矩阵，提高图片质量
        pix = page.get_pixmap(matrix=mat) # 渲染为像素图
        
        # 转换为PIL Image
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data))
        
        # 将图片转换为base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 关闭文档
        doc.close()
        
        return f"data:image/png;base64,{img_base64}"
    
    except Exception as e:
        print(f"解析PDF失败: {e}")
        return None

# 初始化OpenAI客户端
client = OpenAI(
    # 如果没有配置环境变量，请用百炼API Key替换：api_key="sk-xxx"
    api_key = "sk-0ad166a422184f00aa7338de03abd122",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复
# enable_thinking = False # 关闭思考过程

PDF_FILE_PATH = "E:\\study_video\\large_model\\八斗学院\\第10周：多模态大模型\\应阔浩-2025自如企业级AI架构落地的思考与实践.pdf"
 
# 把PDF第一页转成base64图片
pdf_image_base64 = pdf_first_page_to_image(PDF_FILE_PATH)
if not pdf_image_base64:
    exit("PDF转换失败")
    
# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen-vl-max",  # 正确模型名
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": pdf_image_base64}  # 这里是 url，绝对正确
                },
                {"type": "text", "text": "请详细解析这一页PDF的内容,包括文字、图表、结构等。"}
            ]
        }
    ],
    stream=True
    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
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

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)