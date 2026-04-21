
import os
from openai import OpenAI


api_key_str = "sk-c9b1982f0e674957ba9da72ce95922d6"

client = OpenAI(
    # 2. 这里配置阿里云的官方地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key=api_key_str,
)

try:
    print("正在向阿里云发送请求...")
    completion = client.chat.completions.create(
        model="qwen-plus",  # 也可以试 qwen-turbo
        messages=[
            {'role': 'user', 'content': '你好，请回复“收到”'}
        ]
    )
    # 如果成功，会打印内容
    print("连接成功！回复内容：")
    print(completion.choices[0].message.content)

except Exception as e:
    print("出错了：", e)