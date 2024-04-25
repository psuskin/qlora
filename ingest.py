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

nlp = spacy.load("en_core_web_sm")

texts = [""]
for file in os.listdir("data/en_articles_klio"):
    description = open(f"data/en_articles_klio/{file}", encoding="utf-8").read()

    blocks = ["This is the description of" + block for block in description.split("This is the description of") if block]

    for block in blocks:
        if len(texts[-1].split()) + len(block.split()) < 1000:
            texts[-1] += block
        else:
            if len(block.split()) >= 1000:
                texts.append("")
                doc = nlp(block)
                for sent in doc.sents:
                    if len(texts[-1].split()) + len(sent.text.split()) < 1000:
                        texts[-1] += sent.text
                    else:
                        texts.append(sent.text.strip())
            else:
                texts.append(block.strip())
    texts.append("")
texts = [text for text in texts if text]

chunks = [Document(page_content=text) for text in texts if text]

embeddings = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
                embed_instruction="Represent the document for retrieval:",
                query_instruction="Represent the question for retrieving supporting documents:",
            )

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="DB_KLIO_ALPACA",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )
)