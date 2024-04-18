import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, pipeline
from peft import PeftModel

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)

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
                embed_instruction="Represent the document for retrieval:",
                query_instruction="Represent the question for retrieving supporting documents:",
            )

db = Chroma(persist_directory="DB_KLIO_ALPACA", embedding_function=embeddings, client_settings=Settings(anonymized_telemetry=False, is_persistent=True))
retriever = db.as_retriever()

model, tokenizer = load_model(model_id, 'output/klio-alpaca-2-13b-r64-noeval/checkpoint-1875/adapter_model')
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

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # # Print the relevant sources used for the answer
        print("----------------------------------SOURCE DOCUMENTS---------------------------")
        for document in docs:
            print(document.page_content)
        print("----------------------------------SOURCE DOCUMENTS---------------------------")

def infer():
    query = "What is a gozintograph?"

    response = qa(query)
    answer, docs = response["result"], response["source_documents"]

    # Print the result
    print("\n\n> Question:")
    print(query)
    print("\n> Answer:")
    print(answer)

    # # Print the relevant sources used for the answer
    print("----------------------------------SOURCE DOCUMENTS---------------------------")
    for document in docs:
        print(document.page_content)
    print("----------------------------------SOURCE DOCUMENTS---------------------------")

if __name__ == '__main__':
    infer()