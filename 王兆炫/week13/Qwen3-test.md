首先要下载Qwen 0.6B, 下载后model在jupyter-notebook工作目录下, Qwen-0.6B文档内

包含 `safetensors` 和 `tokenizer` 两个文件

<img width="862" height="177" alt="image" src="https://github.com/user-attachments/assets/28daf722-b2a0-40ce-baab-e91efa580c59" />

下载后即可调用model, 完成回答用户的提问

1. `think` 模式

```bash
<think>
Okay, the user wants a short introduction to large language models.
Let me start by recalling what I know. Large language models are AI systems that can understand and generate human language.
They're trained on massive datasets, so they can learn complex patterns and nuances.

I should mention their ability to understand and generate text, not just specific tasks.
Maybe include examples like chatbots or content generation. Also, emphasize their adaptability and efficiency.
Oh, and maybe touch on their applications in various fields. Let me check if I'm covering all key points without being too technical.
 Keep it concise, around 3-4 sentences. Make sure it's clear and easy to understand.
</think>

Large language models (LLMs) are AI systems designed to understand and generate human language.
They are trained on vast datasets, allowing them to learn complex patterns and nuances,
making them versatile for tasks like writing, answering questions, and even creative content generation.
These models can adapt to new information and provide contextually relevant responses, making them valuable tools across industries.
```

2. 使用提示词关闭 `think`

只需在prompt末尾增加: `/no_think`

```python
prompt = "Give me a short introduction to large language models. /no_think"
```

可以看到此时的输出, `<think>` 标签依旧存在, 但其中为空

```bash
<think>

</think>

A large language model (LLM) is a type of artificial intelligence that can understand, generate, and respond to human language.
These models are trained on vast amounts of text data to learn patterns and understand context,
allowing them to perform a wide range of tasks, from writing text to answering questions.
```

3. 使用配置关闭 `think`

修改 `add_thinking` 参数
```python
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=USE_REASONING_MODEL
)
```
修改为
```python
tokenizer = Qwen3Tokenizer(
    tokenizer_file_path=tokenizer_file_path,
    repo_id=repo_id,
    apply_chat_template=True,
    add_generation_prompt=True,
    add_thinking=False
)
```

可以看到此时的输出:
```bash
A large language model (LLM) is a type of artificial intelligence that can understand, generate, and respond to human language.
These models are trained on vast amounts of text data to learn patterns and understand context,
allowing them to perform a wide range of tasks, from writing text to answering questions.
```

可以看到此时是没有 `<think>` 标签的, 因为在加载时就没有开启think









