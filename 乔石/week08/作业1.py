import openai
from pydantic import BaseModel, Field

client = openai.OpenAI(
    api_key="sk-f0172bf78090473ba143e712a4b18ee9",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


class Agent:
    def __init__(self, model_name: str, ):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()["title"],
                    "description": response_model.model_json_schema()["description"],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()["properties"],
                        "required": response_model.model_json_schema()["required"]
                    }
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Translate(BaseModel):
    """文本翻译识别，识别出 原始语种、目标语种，待翻译的文本"""
    original_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text: str = Field(description="待翻译的文本内容")


agent = Agent(model_name="qwen-plus")
result = agent.call(user_prompt="帮我把我今天很开心翻译成英文", response_model=Translate)
print(result)