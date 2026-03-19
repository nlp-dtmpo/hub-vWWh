# test_prompt.py
import openai
import json

# 复制您的提示词（或者从prompt.py导入）
from prompt import PROMPT

# 配置API（需要替换为您的真实API密钥）
client = openai.OpenAI(
    api_key="sk-c9b1982f0e674957ba9da72ce95922d6",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 测试用例
test_cases = [
    "查询北京到上海的高铁票",
    "播放周杰伦的稻香",
    "明天杭州天气怎么样",
    "红烧肉怎么做",
    "打电话给张老师",
    "翻译你好英语怎么说",
    "给我讲个笑话"
]


def test_prompt():
    print("=" * 60)
    print("提示词测试结果")
    print("=" * 60)

    for i, text in enumerate(test_cases, 1):
        print(f"\n【测试{i}】")
        print(f"输入: {text}")

        try:
            response = client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {"role": "system", "content": "你是一个信息抽取专家，只输出JSON格式"},
                    {"role": "user", "content": PROMPT.format(input_text=text)}
                ],
                temperature=0.1
            )

            result = response.choices[0].message.content
            print(f"输出: {result}")
        except Exception as e:
            print(f"错误: {e}")

        print("-" * 40)


if __name__ == "__main__":
    test_prompt()
    print("\n✅ 提示词测试完成！")