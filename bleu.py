from inference import load_model, generate

import os
import re
import json
import torch
import pickle
import statistics
import numpy as np
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD
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

import matplotlib.pyplot as plt
def truncate():
    filename = "alpaca-2-7b-r64-truncated-r8"

    inits = torch.load(os.path.join(modelDir, filename, "init-r64-meta-llama", "Llama-2-7b-hf", "adapter_model.bin"))
    weights = torch.load(os.path.join(modelDir, filename, "checkpoint-1875-r64", "adapter_model", "adapter_model.bin"))

    newWeights = {}
    for name in weights:
        print(name)

        init = inits[name].detach().cpu().float().numpy()
        matrix = weights[name].detach().cpu().float().numpy()

        if 0:
            print(np.linalg.svd(matrix, compute_uv=False))
            exit()

        diff = matrix - init

        U, S, Vt = np.linalg.svd(diff, full_matrices=False)
        print(U.shape, S.shape, Vt.shape)
        diffTrunc = U[:, :8] @ np.diag(S[:8]) @ Vt[:8, :]

        #print(np.sum(diffTrunc - diff)**2)
        #print(np.sum((init + diffTrunc) - matrix)**2)
        newWeights[name] = torch.from_numpy(init + diffTrunc)

    torch.save(newWeights, os.path.join(modelDir, filename, "checkpoint-1875", "adapter_model", "adapter_model.bin"))

def bleuTrunc():
    bleuScores = {"scores": [], "responses": []}

    with open("data/en_articles_alpaca.json", encoding="utf-8") as f:
        data = json.load(f)

    filename = "alpaca-2-7b-r64-truncated-r8"

    modelMatch = re.search(r"-([0-9]+)b-r([0-9]+).*?-r([0-9]+)", filename)
    paramCount = int(modelMatch.group(1))
    rank = int(modelMatch.group(2))
    trunc = int(modelMatch.group(3))

    foundationModelName = f"meta-llama/Llama-2-{paramCount}b-hf"
    _, model, tokenizer = load_model(foundationModelName, os.path.join(modelDir, filename, "checkpoint-1875", "adapter_model"))

    #print(paramCount, rank, trunc)

    for i, sample in enumerate(data):
        if not i % 2:
            continue

        if i > 2 * maxSamples:
            break

        output = generate(model, tokenizer, sample["input"], False)
        target = sample["output"]

        print(output, target)

        bleuScore = sentence_bleu([target], output)
        print(bleuScore)

        bleuScores["scores"].append(bleuScore)
        bleuScores["responses"].append(output)

    with open(f"bleu-trunc-r{rank}-r{trunc}.pickle", "wb") as handle:
        pickle.dump(bleuScores, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved bleu scores")

    print(statistics.mean(bleuScores["scores"]), statistics.stdev(bleuScores["scores"]))

if __name__ == "__main__":
    # bleu()

    #truncate()
    bleuTrunc()