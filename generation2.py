import re
import os
import json
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

def instruct(generations):
    model_path = hf_hub_download(
        repo_id="lightblue/suzume-llama-3-8B-multilingual-gguf",
        filename="ggml-model-Q4_K_M.gguf",
        resume_download=True
    )

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=8192,
        max_tokens=4096,
        n_batch=512,
        n_gpu_layers=100,
        rope_freq_base=500000
    )

    generation_kwargs = {
        "max_tokens": 4096,
        "echo": False,
        "top_k": 1
    }

    for generation in generations:
        lang = generation.get("lang", "en")
        text = generation.get("chunk")
        questions = generation.get("questions", [])
        if not text or not questions:
            continue

        for question in questions:
            if lang == "en":
                prompt = f"""In the following, you will be provided with the description of a module, as well as a query referencing this description which may or may not be answerable based solely on the information provided in the module description. Your task is to assess whether or not the query can be answered with the information provided in the module description. If the query can be answered without additional information, provide the answer. Otherwise, state that the information is not currently available.

Module description: {text}

Query: {question}"""
                llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
            elif lang == "de":
                prompt = f"""Im Folgenden wird Ihnen die Beschreibung eines Moduls sowie eine Abfrage vorgelegt, bei der sich die Abfrage auf die Beschreibung bezieht und nicht unbedingt basierend ausschließlich auf die in der Modulbeschreibung bereitgestellten Informationen beantwortet werden kann. Ihre Aufgabe besteht darin, zu beurteilen, ob die Abfrage mit den in der Modulbeschreibung bereitgestellten Informationen beantwortet werden kann. Wenn die Abfrage ohne zusätzliche Informationen beantwortet werden kann, geben Sie die Antwort an. Andernfalls geben Sie an, dass die Informationen derzeit nicht verfügbar sind.
                
Modulbeschreibung: {text}

Abfrage: {question}"""
                llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Beantworten Sie immer so hilfreich wie möglich, während Sie sicher sind. Wenn Sie die Antwort auf eine Frage nicht kennen, geben Sie bitte keine falschen Informationen weiter.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            if llamaPrompt:
                response = llm(llamaPrompt, **generation_kwargs)

                with open("answers.jsonl", "a") as f:
                    f.write(json.dumps({
                        "lang": lang,
                        "text": text,
                        "question": question,
                        "answer": response
                    }) + "\n")


from langchain.docstore.document import Document
if __name__ == '__main__':
    generations = []
    with open("questions.jsonl", encoding="utf-8") as f:
        for line in f:
            generations.append(json.loads(line))
    instruct(generations)