import json
import pandas as pd

from transformers import (
    AutoTokenizer,
    LlamaTokenizer
)

#model_name_or_path = "huggyllama/llama-7b"
#model_name_or_path = "meta-llama/Llama-2-7b-hf"
model_name_or_path = "meta-llama/Llama-2-13b-hf"
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
    use_auth_token=True,
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
    if isinstance(path, str):
        with open(path, encoding="utf-8") as f:
            jsonArray = json.load(f)
    else:
        jsonArray = path

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
    print(f"Samples under max sequence length: {samplesUnderMaxLength} (approx. {float(samplesUnderMaxLength) / len(tokenCounts)}).")

    samplesOverMaxLength = [jsonArray[i][sequencePart] for i in range(len(tokenCounts)) if tokenCounts[i] > maxTokens]
    #print("Samples over max sequence length: ", samplesOverMaxLength)

    wordCounts = [len(jsonObject[sequencePart].split()) for jsonObject in jsonArray]
    tokensPerWord = [t / w for w, t in zip(wordCounts, tokenCounts)]
    print(f"Average tokens per word: {sum(tokensPerWord) / len(tokensPerWord)}.")

    print()

#count("data/en_articles_autoregressive.json", "output", 2000)

#count("data/en_articles_alpaca.json")

#count("data/en_articles_alpaca.json", "input", 1024)
#count("data/en_articles_alpaca.json", "output", 1024)

#count("data/de_articles_alpaca.json", "input", 1024)
#count("data/de_articles_alpaca.json", "output", 1024)

#count("data/en_articles_klio_alpaca.json", "input", 5120)
#count("data/en_articles_klio_alpaca.json", "output", 5120)

#count("data/en_articles_klio_autoregressive.json", "output", 5120)

from datasets import load_dataset
dataset = load_dataset("timdettmers/openassistant-guanaco")
count(dataset["train"], "text", 5120)

exit()

example = tokenizer(
    #["This evaluation serves the analysis of recorded BDE time tickets."],
    ["inprovis outprovis"],
    max_length=2**20,
    truncation=True,
    add_special_tokens=False,
)

subwords = tokenizer.convert_ids_to_tokens(example["input_ids"][0])

print(subwords)