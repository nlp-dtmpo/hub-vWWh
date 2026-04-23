from openai import OpenAI
import base64

# 初始化客户端
client = OpenAI(
    api_key="your_dashscope_api_key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 图片转Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("your_image.jpg")

# 调用模型
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

# 输出结果
print(response.choices[0].message.content)