from inference import load_model, generate

import os
import re
import json
import statistics
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

modelDir = "/workspace/analysis"

rec_dd = lambda: defaultdict(rec_dd)

def bleu():
    bleuScores = rec_dd()

    with open("data/en_articles_alpaca.json", encoding="utf-8") as f:
        data = json.load(f)

    for filename in os.listdir(modelDir):
        modelMatch = re.search(r"-([0-9]+)b-r([0-9]+)", filename)
        paramCount = int(modelMatch.group(1))
        rank = int(modelMatch.group(2))

        if rank > 7:
            continue

        print(paramCount, rank)

        _, model, tokenizer = load_model(f"meta-llama/Llama-2-{paramCount}b-hf", os.path.join(modelDir, filename, "checkpoint-1875", "adapter_model"))

        for i, sample in enumerate(data):
            if not i % 2:
                continue

            if i > 20:
                break

            output = generate(model, tokenizer, sample["input"], False)
            target = sample["output"]

            #print(output, target)

            bleuScore = sentence_bleu([target], output)
            print(bleuScore)

            if not bleuScores[paramCount][rank]:
                bleuScores[paramCount][rank] = []
            bleuScores[paramCount][rank].append(bleuScore)

    for paramCount in bleuScores:
        for rank in bleuScores[paramCount]:
            print(paramCount, rank, statistics.mean(bleuScores[paramCount][rank]), statistics.stdev(bleuScores[paramCount][rank]))

if __name__ == "__main__":
    bleu()