from huggingface_hub import hf_hub_download
import streamlit as st
import os
from llama_cpp import Llama



select_llm_models = st.selectbox(
    "Download LLM First",
    ("Qwen2.5-1.5B-Instruct","Qwen2.5-3B-Instruct","smollm2-1.7b-instruct","Llama-3.2-1B-Instruct"),
    index=None,
)


models_names = {
    "Qwen2.5-1.5B-Instruct": ["bartowski/Qwen2.5-1.5B-Instruct-GGUF","Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"],

    "Qwen2.5-3B-Instruct": ["bartowski/Qwen2.5-3B-Instruct-GGUF","Qwen2.5-3B-Instruct-Q4_K_M.gguf"],

    "smollm2-1.7b-instruct" : ['HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF',"smollm2-1.7b-instruct-q4_k_m.gguf"],

    "Llama-3.2-1B-Instruct": ["bartowski/Llama-3.2-1B-Instruct-GGUF","Llama-3.2-1B-Instruct-Q4_K_M.gguf"]
}


model_folder_path = ".\model"
download_models = os.listdir(model_folder_path)

if len(download_models) != 0:
    st.write("Downloaded Models:", download_models[1])
    select_model_to_chat = st.selectbox(
          "Select a model to chat with",
          download_models[1:],
          index=None
    )
    if select_model_to_chat != None:
          model_path = ".\model\\" + select_model_to_chat
          llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=0,
            verbose=False
        )

          inp = st.text_area("Enter your prompt here", height=200)
          if inp:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": inp}
                ],
                max_tokens=150,
                temperature=0.3
            )

            st.text_area("Response", value=response['choices'][0]['message']['content'], height=200)



if select_llm_models == None:
        st.warning("Please select a model to download.")
else:
        print(select_llm_models)
    
if select_llm_models in models_names.keys():
        if models_names[select_llm_models][1] in download_models:
            st.write("Model already downloaded:", select_llm_models)
        else:
            model_path = hf_hub_download(
                repo_id=models_names[select_llm_models][0],
                filename=models_names[select_llm_models][1],
                local_dir=r".\model",
            )
            