import json
import pandas as pd

from transformers import (
    AutoTokenizer,
    LlamaTokenizer
)

model_name_or_path = "huggyllama/llama-7b"
cache_dir = None

DEFAULT_PAD_TOKEN = "[PAD]"

max_sequence_length = 2048

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    cache_dir=cache_dir,
    padding_side="right",
    use_fast=False, # Fast tokenizer giving issues.
    tokenizer_type='llama' if 'llama' in model_name_or_path else None, # Needed for HF name change
    trust_remote_code=False,
    use_auth_token=False,
)
if tokenizer._pad_token is None:
    tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))
if 'llama' in model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
    # LLaMA tokenizer may not have correct special tokens set.
    # Check and add them if missing to prevent them from being parsed into different tokens.
    # Note that these are present in the vocabulary.
    # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
    print('Adding special tokens.')
    tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(2),
            "bos_token": tokenizer.convert_ids_to_tokens(1),
            "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
    })

def count(path, sequencePart="output", maxTokens=max_sequence_length):
    with open(path, encoding="utf-8") as f:
        jsonArray = json.load(f)

    sources = [f"{tokenizer.bos_token}{jsonObject[sequencePart]}{tokenizer.eos_token}" for jsonObject in jsonArray]

    tokenized_sources_with_prompt = tokenizer(
        sources,
        max_length=2**20,
        truncation=True,
        add_special_tokens=False,
    )

    print(list(tokenized_sources_with_prompt.keys()))

    tokenCounts = [len(input_ids) for input_ids in tokenized_sources_with_prompt["input_ids"]]
    s = pd.Series(tokenCounts)
    print(s.describe())

    samplesUnderMaxLength = sum(1 for tokenCount in tokenCounts if tokenCount < maxTokens)
    print(f"Samples under max sequence length: {samplesUnderMaxLength} (approx. {float(samplesUnderMaxLength) / len(tokenCounts)}).\n")

    samplesOverMaxLength = [jsonArray[i][sequencePart] for i in range(len(tokenCounts)) if tokenCounts[i] > maxTokens]
    print("Samples over max sequence length: ", samplesOverMaxLength)

#count("data/en_articles_autoregressive.json")

#count("data/en_articles_alpaca.json")

#count("data/en_articles_alpaca.json", "input", 1024)
#count("data/en_articles_alpaca.json", "output", 1024)

count("data/de_articles_alpaca.json", "input", 1024)
count("data/de_articles_alpaca.json", "output", 1024)