import re
import os
import json
from huggingface_hub import hf_hub_download
from langchain.llms import LlamaCpp

def instruct(chunks):
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

    pattern = re.compile(r'\d+\.\s(.+?)(?:\n|$)')
    for chunk in chunks:
        lang = chunk.metadata.get("lang", "en")
        text = chunk.page_content
        if not text:
            continue

        if lang == "en":
            prompt = f"""In the following, you will be provided with the description of a module. Your task is to generate a numbered list of realistic questions referencing this module description from the perspective of an unfamiliar user who would like to know more about a certain functionality specific to the provided module. Please use both the imperative and interrogative forms and try not to repeat verbs for the questions to maximize variety.

If the information you plan to query seems module-specific, please reference the module name in the query. NEVER REFER TO A MODULE IN A QUESTION WITHOUT USING ITS NAME. This means to never use the word "module" in a question if you are not providing the module name. For example, NEVER use the terms "the module" or "this module" in a question.

ONLY GENERATE QUESTIONS WHICH CAN BE ANSWERED SOLELY USING THE MODULE DESCRIPTION. DO NOT ASK QUESTIONS WHICH REQUIRE ADDITIONAL INFORMATION.

Module description: {text}"""
            llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        elif lang == "de":
            prompt = f"""Im Folgenden wird die Beschreibung eines Moduls bereitgestellt. Ihre Aufgabe besteht darin, eine nummerierte Liste realistischer Fragen zu generieren, die sich auf die Modulbeschreibung beziehen, aus der Perspektive eines unbekannten Benutzers, der mehr über eine bestimmte Funktionalität erfahren möchte, die spezifisch für das bereitgestellte Modul ist. Bitte verwenden Sie sowohl die Imperativ- als auch die Interrogativformen und versuchen Sie, Verben für die Fragen nicht zu wiederholen, um die Vielfalt zu maximieren.

Wenn die von Ihnen geplante Abfrage modulspezifisch erscheint, beziehen Sie sich bitte auf den Modulnamen in der Abfrage. BEZIEHEN SIE SICH NIEMALS AUF EIN MODUL IN EINER FRAGE, OHNE SEINEN NAMEN ZU VERWENDEN. Dies bedeutet, dass Sie das Wort "Modul" in einer Frage niemals verwenden, wenn Sie den Modulnamen nicht angeben. Verwenden Sie beispielsweise NIEMALS die Begriffe "das Modul" oder "dieses Modul" in einer Frage.

GENERIEREN SIE NUR FRAGEN, DIE AUSSCHLIESSLICH MIT DER MODULBESCHREIBUNG BEANTWORTET WERDEN KÖNNEN. STELLEN SIE KEINE FRAGEN, DIE ZUSÄTZLICHE INFORMATIONEN BENÖTIGEN.

Modulbeschreibung: {text}"""
            llamaPrompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSie sind ein hilfsbereiter, respektvoller und ehrlicher Assistent. Beantworten Sie immer so hilfreich wie möglich, während Sie sicher sind. Wenn Sie die Antwort auf eine Frage nicht kennen, teilen Sie bitte keine falschen Informationen.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            llamaPrompt = None

        if llamaPrompt:
            response = llm(llamaPrompt, **generation_kwargs)

            questions = pattern.findall(response)

            with open("questions.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"chunk": text, "questions": questions, "lang": lang}))
                f.write("\n")


from langchain.docstore.document import Document
if __name__ == '__main__':
    chunks = [
        Document(page_content='Dies ist die Beschreibung des Moduls "about" mit dem Namen "Systemversion": Anzeige und Auflistung der Versions- und Urheberrechtsinformationen. Dies ist die Beschreibung der Funktionalität des Moduls "about" mit dem Namen "Systemversion" bezüglich Eingabefenster: Dieses Fenster dient der Anzeige der Versions- und Urheberrechtsinformationen.', metadata={'lang': 'de'}),
        Document(page_content='This is the description of the module "about" with the name "System version": Display and listing of version and copyright information. This is the description of the functionality of the module "about" with the name "System version" regarding Input window: This window is used to display the version and copyright information.', metadata={'lang': 'en'}),
    ]
    instruct(chunks)