## Imports
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

## Download the GGUF model
model_path = hf_hub_download(
    repo_id="lightblue/suzume-llama-3-8B-multilingual-gguf",
    filename="ggml-model-Q4_K_M.gguf",
    resume_download=True
)

## Instantiate model from downloaded file
llm = LlamaCpp(
    model_path=model_path,
    n_ctx=8192,
    max_tokens=4096,
    n_batch=512,
    n_gpu_layers=100
)

## Generation kwargs
generation_kwargs = {
    "max_tokens":4096,
    "echo":False, # Echo the prompt in the output
    "top_k":1 # This is essentially greedy decoding, since the model will always return the highest-probability token. Set this value > 1 for sampling decoding
}

## Run inference
prompt = "Was ist der Sinn des Lebens?"
# https://huggingface.co/lightblue/suzume-llama-3-8B-multilingual/blob/c7b55e87c44c7e8d52ead657715c14abd3f9cda9/tokenizer_config.json#L2052
llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
res = llm(llamaPrompt, **generation_kwargs)
print(res)