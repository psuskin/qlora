from inference import load_model, generate

import os
import re
import json
import pickle
import statistics
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu

modelDir = "/home/psuskin/repos/analysis"

rec_dd = lambda: defaultdict(rec_dd)

# TODO: change this for param count to test
paramCountToEvaluate = 7

maxSamples = 1000

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def bleu():
    bleuScores = rec_dd()
    foundationModels = set()

    with open("data/en_articles_alpaca.json", encoding="utf-8") as f:
        data = json.load(f)

    for filename in os.listdir(modelDir):
        modelMatch = re.search(r"-([0-9]+)b-r([0-9]+)", filename)
        paramCount = int(modelMatch.group(1))
        rank = int(modelMatch.group(2))

        if paramCount != paramCountToEvaluate:
            continue

        foundationModelName = f"meta-llama/Llama-2-{paramCount}b-hf"
        foundationModels.add(foundationModelName)
        _, model, tokenizer = load_model(foundationModelName, os.path.join(modelDir, filename, "checkpoint-1875", "adapter_model"))

        print(paramCount, rank)

        for i, sample in enumerate(data):
            if not i % 2:
                continue

            if i > 2 * maxSamples:
                break

            output = generate(model, tokenizer, sample["input"], False)
            target = sample["output"]

            #print(output, target)

            bleuScore = sentence_bleu([target], output)
            print(bleuScore)

            if not bleuScores[paramCount][rank]:
                bleuScores[paramCount][rank] = []
            bleuScores[paramCount][rank].append(bleuScore)

    for foundationModelName in foundationModels:
        paramCount = int(re.search(r"-([0-9]+)b", foundationModelName).group(1))

        if paramCount != paramCountToEvaluate:
            continue

        print(paramCount, "foundation")

        model, _, tokenizer = load_model(foundationModelName, None)

        for i, sample in enumerate(data):
            if not i % 2:
                continue

            if i > 2 * maxSamples:
                break

            output = generate(model, tokenizer, sample["input"], False)
            target = sample["output"]

            #print(output, target)

            bleuScore = sentence_bleu([target], output)
            print(bleuScore)

            if not bleuScores[paramCount][0]:
                bleuScores[paramCount][0] = []
            bleuScores[paramCount][0].append(bleuScore)

    bleu_dict = ddict2dict(bleuScores)
    with open("bleu.pickle", "wb") as handle:
        pickle.dump(bleu_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved bleu scores")

    for paramCount in bleuScores:
        for rank in bleuScores[paramCount]:
            print(paramCount, rank, statistics.mean(bleuScores[paramCount][rank]), statistics.stdev(bleuScores[paramCount][rank]))

if __name__ == "__main__":
    bleu()