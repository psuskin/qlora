import re
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

PATH = "/workspace/output"

USE_CPU = False

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

def plotDistribution(tensor):
    matrix = tensor.detach().cpu().float().numpy()

    plt.hist(matrix.flatten(), bins=100)
    plt.show()

def ensureImageSubset(dirOrig, dirTrans):
    if USE_CPU:
        weightsOrig = torch.load(dirOrig, map_location="cpu")
        weightsTrans = torch.load(dirTrans, map_location="cpu")
    else:
        weightsOrig = torch.load(dirOrig)
        weightsTrans = torch.load(dirTrans)

    weightOrig = next(iter(weightsOrig.values()))
    weightTrans = next(iter(weightsTrans.values()))

    #for row in weightOrig: print(row)
    #for row in weightTrans: print(row)

    print(torch.all(torch.isin(weightTrans, weightOrig)))

    exit()

if __name__ == '__main__':
    #ensureImageSubset("output/init-r64-EleutherAI/pythia-12b/adapter_model.bin", "output/init-r32-EleutherAI/pythia-12b/adapter_model.bin")

    weights = {}

    getInitWeights(weights)
    #plotDistribution(weights[64]["init"]['base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight'])

    models = ["alpaca-2-7b/checkpoint-1875", "alpaca-2-7b-r32/checkpoint-1875", "alpaca-2-7b-r16/checkpoint-1875"]
    getWeights(weights, models)

    SVDs = getSVDs(weights)