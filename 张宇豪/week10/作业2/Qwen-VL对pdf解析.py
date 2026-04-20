import base64
import io
from openai import OpenAI
from pdf2image import convert_from_path

client = OpenAI(
    api_key='sk-08a6acbb4a2e46e195b8199036824588',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
)

def analyze_pdf_first_page(pdf_file):
    print('正在读取PDF...')
    images = convert_from_path(pdf_file, first_page=1, last_page=1, dpi=200)

    buffered = io.BytesIO()
    images[0].save(buffered, format='JPEG')
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    print('正在请求云端模型...')
    response = client.chat.completions.create(
        model='qwen-vl-max',
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细解析这个 PDF 文件第一页的内容，以 Markdown 格式输出关键信息。"},
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

    return response.choices[0].message.content

if __name__ == '__main__':
    path = r"D:\八斗学院\第6周：RAG工程化实现\Week06\汽车知识手册.pdf"
    try:
        result = analyze_pdf_first_page(path)
        print("------解析结果------")
        print(result)
    except Exception as e:
        print(f"发生错误：{e}")
