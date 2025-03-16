# pip install llama-cpp-python

from llama_cpp import Llama

# 初始化模型
llm = Llama(
    model_path="gemma3_1b.gguf",
    n_ctx=512,
    chat_format="simple",
    seed=42,
    n_batch=1,
    n_ubatch=1,
)

# 之前的对话历史
conversation_history = [
    "User: how are you?"
]

# 将对话历史和“Assistant:”结合，作为输入
input_text = ("""User: how are you?\n
              Assistant:""")

# 获取模型的输出
output = llm(input_text, max_tokens=15)

# 从模型输出中提取助手的回复
assistant_reply = output["choices"][0]["text"].strip()

# 打印助手的回答
print(assistant_reply)

