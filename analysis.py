import re
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

PATH = "/workspace/analysis"

USE_CPU = True

rec_dd = lambda: defaultdict(rec_dd)

class Weights:
    def __init__(self, tensor):
        self.matrix = tensor.detach().cpu().float().numpy()

        #self.U, self.S, self.Vh = np.linalg.svd(self.matrix)

class Layer:
    def __init__(self):
        self.modules = rec_dd()

class Model:
    def __init__(self, directory):
        if rankMatch := re.search(r"-r([0-9]+)", directory):
            self.rank = int(rankMatch.group(1))
        else:
            self.rank = 64

        if modelMatch := re.search(r"(.*?)-r", directory):
            self.foundation = modelMatch.group(1)
        else:
            self.foundation = directory


        if initBinaryFound := glob.glob(f"{directory}/init-*/*/adapter_model.bin"):
            if os.path.exists(resultBinary := os.path.join(directory, "checkpoint-1875", "adapter_model", "adapter_model.bin")):
                self.layers = self._loadLayers(initBinaryFound[0], resultBinary)

    def _loadLayers(self, initBinary, resultBinary):
        layers = defaultdict(Layer)

        if USE_CPU:
            initTorchDir = torch.load(initBinary, map_location="cpu")
            resultTorchDir = torch.load(resultBinary, map_location="cpu")
        else:
            initTorchDir = torch.load(initBinary)
            resultTorchDir = torch.load(resultBinary)

        for module in initTorchDir:
            layerIndex, layerFragment, layerMatrix = self._getRegex(module)
            layers[int(layerIndex)].modules[layerFragment][layerMatrix]["init"] = Weights(initTorchDir[module])

        for module in resultTorchDir:
            layerIndex, layerFragment, layerMatrix = self._getRegex(module)
            layers[int(layerIndex)].modules[layerFragment][layerMatrix]["result"] = Weights(resultTorchDir[module])

        return layers
    
    def _getRegex(self, module):
        layerMatch = re.search(r"layers.([0-9]+).(.*?).lora_([A-B]).", module)
        return layerMatch.group(1), layerMatch.group(2), layerMatch.group(3)

def grassmann(A, B, i, j, SVD=True):
    if SVD:
        _, _, AVh = np.linalg.svd(A.matrix)
        _, _, BVh = np.linalg.svd(B.matrix)
    else:
        AVh = A
        BVh = B
    
    Ui = AVh.T[:, :i]
    Uj = BVh.T[:, :j]

    return np.linalg.norm(Ui.T @ Uj)**2 / min(i, j)

def plotDistribution(matrix):
    plt.hist(matrix.flatten(), bins=100)
    plt.show()

def ensureImageSubset(dirOrig, dirTrans):
    if USE_CPU:
        weightsOrig = torch.load(dirOrig, map_location="cpu")
        weightsTrans = torch.load(dirTrans, map_location="cpu")
    else:
        weightsOrig = torch.load(dirOrig)
        weightsTrans = torch.load(dirTrans)

    weightOrig = next(iter(weightsOrig.values())).detach().cpu().float()
    weightTrans = next(iter(weightsTrans.values())).detach().cpu().float()

    #for row in weightOrig: print(row)
    #for row in weightTrans: print(row)

    print(torch.all(torch.isin(weightTrans, weightOrig)))

    exit()

specificModels = ["alpaca-2-7b-r64", "alpaca-2-7b-r8"]

if __name__ == '__main__':
    #ensureImageSubset(os.path.join(PATH, "alpaca-2-13b-r64/init-r64-meta-llama/Llama-2-13b-hf/adapter_model.bin"), os.path.join(PATH, "/workspace/analysis/alpaca-2-13b-r32/init-r32-meta-llama/Llama-2-13b-hf/adapter_model.bin"))

    models = {}
    for directory in os.listdir(PATH):
        if not specificModels or directory in specificModels:
            models[directory] = Model(os.path.join(PATH, directory))

    #print(models)
    #print(models["alpaca-2-7b-r64"].layers)
    #print(models["alpaca-2-7b-r64"].layers[0])
    #print(models["alpaca-2-7b-r64"].layers[0].modules)
    #print(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"])

    #print(grassmann(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"], models["alpaca-2-7b-r32"].layers[0].modules["self_attn.q_proj"]["A"]["init"], 16, 8))
    #print(grassmann(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["result"], models["alpaca-2-7b-r32"].layers[0].modules["self_attn.q_proj"]["A"]["result"], 16, 8))

    _, _, AVh = np.linalg.svd(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["result"].matrix)
    _, _, BVh = np.linalg.svd(models["alpaca-2-7b-r8"].layers[0].modules["self_attn.q_proj"]["A"]["result"].matrix)
    grassmann_matrix = np.zeros((64, 8))
    for j in range(1, 8+1):
        for i in range(j, 64+1):
            dist = grassmann(AVh, BVh, i, j, False)
            print(dist)
            grassmann_matrix[i-1, j-1] = dist
    
    fix, ax = plt.subplots()

    ax.matshow(grassmann_matrix, cmap=plt.cm.Blues)
    for i in range(64):
        for j in range(8):
            c = grassmann_matrix[i, j]
            #ax.text(i, j, str(c), va='center', ha='center')

    plt.savefig("figure.png")

    #plotDistribution(models["alpaca-2-7b-r64"].layers[0]["self_attn.q_proj"]["A"]["init"])