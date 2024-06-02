import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from peft import PeftModel

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import re
import json

#EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

device_type = "cuda" if torch.cuda.is_available() else "cpu"

model_id = 'meta-llama/Llama-2-13b-hf'

def load_model(model_name_or_path='huggyllama/llama-7b', adapter_path=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # Fixing some of the early LLaMA HF conversion issues.
    tokenizer.bos_token_id = 1

    # Load the model (use bf16 for faster inference)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )
    )

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
    else:
        model = base_model

    return model, tokenizer

embeddings = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
                embed_instruction="query:",
                query_instruction="passage:",
            )

db = Chroma(persist_directory="DB", embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False, is_persistent=True))
retriever = db.as_retriever(search_kwargs={"k": 2})

model, tokenizer = load_model(model_id, 'ADAPTER_NEW/checkpoint-1875/adapter_model')
generation_config = GenerationConfig.from_pretrained(model_id)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=4096,
    temperature=0.2,
    # top_p=0.95,
    repetition_penalty=1.15,
    generation_config=generation_config,
    # exponential_decay_length_penalty=(15, 1.01),
)
llm = HuggingFacePipeline(pipeline=pipe)

template = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{question}\n\n###Input:\n{context}\n\n### Response:"
prompt = PromptTemplate(input_variables=["history", "context", "question"], template=template)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": prompt,
    },
)

def run():
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        response = qa(query)
        answer, docs = response["result"], response["source_documents"]
        answer = answer.split("### Response:", 1)[1].strip()
        answer.replace("SAP", "classix")

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print(re.search(r'"([^"]+)"', document.page_content).group(1))
            #print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

def infer():
    responses = []

    EN = False

    if EN:
        queries = [
            "What is a variant part?",
            "What is a gozintograph?",
            "How do I plan production orders?",
            "What are specification numbers?",
            "What is a subject characteristics bar?",
            "How do I implement attributes into subject characteristics bars?",
            "How are parts evaluated in the warehouse?",
            "What is a price table?",
            "What are conditional parts list items?",
            "How do I import my inventory data into GESTIN?",
            "What is inventory sampling?",
            "What does PYTHIA do?",
            "Can I change the output currency of an order confirmation?",
            "What is a packing list?",
            "What is master data?"
        ]
    else:
        queries = [
            "Was ist ein Variantenteil?",
            "Was ist ein Gozintograph?",
            "Wie plane ich Fertigungsaufträge?",
            "Was sind Spezifikationsnummern?",
            "Was ist eine Sachmerkmalsleiste?",
            "Wie implementiere ich Attribute in Merkmalsleisten?",
            "Wie werden Teile im Lager bewertet?",
            "Was ist eine Preisliste?",
            "Was sind bedingte Teilelistenpositionen?",
            "Wie importiere ich meine Lagerdaten in GESTIN?",
            "Was ist Stichprobeninventur?",
            "Was macht PYTHIA?",
            "Kann ich die Ausgabewährung einer Auftragsbestätigung ändern?",
            "Was ist ein Lieferschein?",
        ]

    for query in queries:
        response = qa(query)
        answer, docs = response["result"], response["source_documents"]

        #from saliency import runOnOutput
        #subwords, attributed = runOnOutput(model, tokenizer, answer.split("### Response:")[0] + "### Response:", answer)

        #answer = answer.split("### Response:", 1)[1].strip()
        answer.replace("SAP", "classix")

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        modules = []
        for document in docs:
            module = re.search(r'"([^"]+)"', document.page_content).group(1)
            if module not in modules:
                modules.append(module)
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

        responses.append({"query": query, "response": answer, "docs": [doc.page_content for doc in docs], "modules": modules})

    with open("ADAPTER_NEW/evaluation.json", "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    #run()
    infer()