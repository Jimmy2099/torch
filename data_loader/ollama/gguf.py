
from llama_cpp import Llama

llm = Llama(
    model_path="gemma3_1b.gguf",
    n_ctx=512,
    chat_format="simple",
    seed=42,
    n_batch=1,
    n_ubatch=1,
)

conversation_history = [
    "User: how are you?"
]

input_text = ("""User: how are you?\n
              Assistant:""")

output = llm(input_text, max_tokens=15)

assistant_reply = output["choices"][0]["text"].strip()

print(assistant_reply)

