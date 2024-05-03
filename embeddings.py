import re
import os
import json
import math
import torch
import spacy

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)

device_type = "cuda" if torch.cuda.is_available() else "cpu"

templates = {
    "en": {
        "model": spacy.load("en_core_web_sm"),
        "split": "This is the description"
    },
    "de": {
        "model": spacy.load("de_core_news_sm"),
        "split": "Dies ist die Beschreibung"
    }
}


def createDB(path, dirs):
    chunks = []

    texts = [""]
    for dir in dirs:
        lang = os.path.basename(os.path.normpath(dir)).split("_", 1)[0]
        for file in os.listdir(dir):
            description = open(os.path.join(dir, file), encoding="utf-8").read()

            blocks = [templates[lang]["split"] + block for block in description.split(templates[lang]["split"]) if block]

            for block in blocks:
                if len(texts[-1].split()) + len(block.split()) < 1000:
                    texts[-1] += block
                else:
                    if len(block.split()) >= 1000:
                        texts.append("")
                        doc = templates[lang]["model"](block)
                        for sent in doc.sents:
                            if len(texts[-1].split()) + len(sent.text.split()) < 1000:
                                texts[-1] += sent.text
                            else:
                                texts.append(sent.text.strip())
                    else:
                        texts.append(block.strip())
            texts.append("")

        chunks.extend([Document(page_content=text, metadata={"origin": dir}) for text in texts if text])

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        embed_instruction="Represent the document for retrieval:",
        query_instruction="Represent the question for retrieving supporting documents:",
    )

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=path,
        client_settings=Settings(
            anonymized_telemetry=False,
            is_persistent=True,
        )
    )


def createVectorSpace(path):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device_type},
        embed_instruction="Represent the document for retrieval:",
        query_instruction="Represent the question for retrieving supporting documents:",
    )

    db = Chroma(persist_directory=path, embedding_function=embeddings,
                client_settings=Settings(anonymized_telemetry=False, is_persistent=True))

    data = db.get(include=['embeddings', 'documents', 'metadatas'])

    with open("embeddings/chunks.json", "w", encoding="utf-8") as f:
        json.dump(data, f)

if __name__ == '__main__':
    createDB("embeddings/DB", ["embeddings/appswarehouse.de/en_rule", "embeddings/instantview.org/en_rule"])
    createVectorSpace("embeddings/DB")