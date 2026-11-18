
from llama_cpp import Llama

llm = Llama(
    model_path=r"D:\Hamza Ali\ALL PYTHON CODE DO HERE PLEASE\FYP OFFLINE RAG\models\qwen2.5-1.5b-instruct-q3_k_m.gguf",
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)

response = llm.create_chat_completion(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer accurately and simply."
        },
        {
            "role": "user",
            "content": "write application for sick leave school name AL Haram Academy to my principal"
        }
    ],
    temperature=0.3,
    max_tokens=150
)

print(response["choices"][0]["message"]["content"])
