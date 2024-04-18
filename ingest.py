import re
import json
import torch

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from chromadb.config import Settings

EMBEDDING_MODEL_NAME = "hkunlp/instructor-large"  # Uses 1.5 GB of VRAM (High Accuracy with lower VRAM usage)

device_type = "cuda" if torch.cuda.is_available() else "cpu"

with open("data/en_articles_klio_alpaca.json", encoding="utf-8") as f:
    samples = json.load(f)

blocks = []
contexts = set()
for sample in samples:
    context = re.search(r"### Input:\n(.*)\n\n### Response:", sample["input"]).group(1)
    if context in contexts:
        continue

    contexts.add(context)
    blocks.append(Document(page_content=context))

embeddings = HuggingFaceInstructEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": device_type},
                embed_instruction="Represent the document for retrieval:",
                query_instruction="Represent the question for retrieving supporting documents:",
            )

db = Chroma.from_documents(
    blocks,
    embeddings,
    persist_directory="DB_KLIO_ALPACA",
    client_settings=Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )
)