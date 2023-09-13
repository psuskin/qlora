import re
import os
import torch
import numpy as np

PATH = "/workspace/output"

USE_CPU = True

RANKS = [64, 32, 16]

def getInitWeights(weights):
    for rank in RANKS:
        weights[rank] = {}
        if USE_CPU:
            weights[rank]["init"] = torch.load(f"init-r{rank}/adapter_model.bin", map_location="cpu")
        else:
            weights[rank]["init"] = torch.load(f"init-r{rank}/adapter_model.bin")

def getWeights(weights, models):
    for model in models:
        result = re.search(r"-r([0-9]+)", model)
        if result:
            rank = int(result.group(1))
        else:
            rank = 64

        if rank not in weights:
            raise Exception(f"Rank {rank} identified for {model} does not have initial weight matrices defined. Initial weight matrices are defined for the following ranks: {sorted(list(weights.keys()))}")

        if USE_CPU:
            weights[rank][model] = torch.load(os.path.join(PATH, model, "adapter_model/adapter_model.bin"), map_location="cpu")
        else:
            weights[rank][model] = torch.load(os.path.join(PATH, model, "adapter_model/adapter_model.bin"))

def getSVDs(weights):
    SVDs = {}
    for rank in weights:
        SVDs[rank] = {}
        for model in weights[rank]:
            for module in weights[rank][model]:
                tensor = weights[rank][model][module]
                matrix = tensor.detach().cpu().float().numpy()
                U, S, Vh = np.linalg.svd(matrix)
                print(U, S, Vh)
                exit()

    return SVDs

if __name__ == '__main__':
    weights = {}

    getInitWeights(weights)

    models = ["alpaca-2-7b/checkpoint-1875", "alpaca-2-7b-r32/checkpoint-1875", "alpaca-2-7b-r16/checkpoint-1875"]
    getWeights(weights, models)

    initSVDs = getSVDs(weights)