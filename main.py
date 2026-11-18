from huggingface_hub import hf_hub_download
import streamlit as st
import os


select_llm_models = st.selectbox(
    "Download LLM First",
    ("Qwen 2.5 -- 1b Parameters","Qwen2.5 3B Instruct","SmolLM2 1.7b","Llama 3.2 1b"),
    index=None,
)

print(select_llm_models)

if select_llm_models == None:
    st.warning("Please select a model to download.")
else:
    print(select_llm_models)
    st.write("very good")
    

models_names = {
    "Qwen 2.5 -- 1b Parameters": ["bartowski/Qwen2.5-1.5B-Instruct-GGUF","Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"],

    "Qwen2.5 3B Instruct": ["bartowski/Qwen2.5-3B-Instruct-GGUF","Qwen2.5-3B-Instruct-Q4_K_M.gguf"],

    "SmolLM2 1.7b" : ['HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF',"smollm2-1.7b-instruct-q4_k_m.gguf"],

    "Llama 3.2 1b": ["bartowski/Llama-3.2-1B-Instruct-GGUF","Llama-3.2-1B-Instruct-Q4_K_M.gguf"]
}

if select_llm_models in models_names.keys():
    model_path = hf_hub_download(
        repo_id=models_names[select_llm_models][0],
        filename=models_names[select_llm_models][1],
        local_dir=r".\model",
    )

model_folder_path = ".\model"
download_models = os.listdir(model_folder_path)

if len(download_models) != 0:
    st.write("Downloaded Models:", download_models[1])

# model_path = hf_hub_download(
#     repo_id=,
#     filename=,
#     local_dir=r".\model",
# )

# print("Downloaded:", model_path)