
import psutil
import time
import threading
from llama_cpp import Llama

process = psutil.Process()
peak_ram = 0
running = True

def monitor_ram():
    global peak_ram

    while running:
        ram = process.memory_info().rss / (1024 ** 3)
        peak_ram = max(peak_ram, ram)
        time.sleep(0.1)

# RAM before loading
before = process.memory_info().rss / (1024 ** 3)

monitor = threading.Thread(target=monitor_ram)
monitor.start()

without_moe_model = r"D:\Hamza Ali\ALL PYTHON CODE DO HERE PLEASE\FYP OFFLINE RAG\models\qwen2.5-1.5b-instruct-q3_k_m.gguf"

moe_model = r"D:\Hamza Ali\ALL PYTHON CODE DO HERE PLEASE\FYP OFFLINE RAG\model\Qwen2-1.5Moe.IQ3_XS.gguf"

# Load model
llm = Llama(
    model_path=without_moe_model,
    n_ctx=2048,
    n_gpu_layers=0,
    verbose=False
)

after_load = process.memory_info().rss / (1024 ** 3)

# Generate answer
start = time.perf_counter()

response = llm.create_chat_completion(
    messages=[
        {"role": "user", "content": "Explain artificial intelligence."}
    ],
    max_tokens=150,
    temperature=0.3
)

elapsed = time.perf_counter() - start

running = False
monitor.join()

print("\n--- MODEL BENCHMARK ---")
print(f"RAM before loading: {before:.2f} GB")
print(f"RAM after loading: {after_load:.2f} GB")
print(f"Peak RAM usage: {peak_ram:.2f} GB")
print(f"RAM used by model loading: {after_load-before:.2f} GB")
print(f"Generation time: {elapsed:.2f} seconds")
print(f"Peak process RAM: {peak_ram:.2f} GB")