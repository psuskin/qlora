import re
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

PATH = "/workspace/analysis"

USE_CPU = True

RANKS = [64, 32, 16]

rec_dd = lambda: defaultdict(rec_dd)

class Weights:
    def __init__(self, tensor):
        self.matrix = tensor.detach().cpu().float().numpy()

        self.U, self.S, self.Vh = np.linalg.svd(self.matrix)

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
            layers[layerIndex].modules[layerFragment][layerMatrix]["init"] = Weights(initTorchDir[module])

        for module in resultTorchDir:
            layerIndex, layerFragment, layerMatrix = self._getRegex(module)
            layers[layerIndex].modules[layerFragment][layerMatrix]["result"] = Weights(initTorchDir[module])
    
    def _getRegex(self, module):
        layerMatch = re.search(r"layers.([0-9]+).(.*?).lora_([A-B]).", module)
        return layerMatch.group(1), layerMatch.group(2), layerMatch.group(3)

def grassmann(matrixLarge, matrixSmall):
    pass

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

if __name__ == '__main__':
    #ensureImageSubset(os.path.join(PATH, "alpaca-2-13b-r64/init-r64-meta-llama/Llama-2-13b-hf/adapter_model.bin"), os.path.join(PATH, "/workspace/analysis/alpaca-2-13b-r32/init-r32-meta-llama/Llama-2-13b-hf/adapter_model.bin"))

    models = {}
    for directory in os.listdir(PATH):
        models[directory] = Model(os.path.join(PATH, directory))

    #print(models)
    print(models["alpaca-2-7b-r64"].layers[0])

    #plotDistribution(models["alpaca-2-7b-r64"].layers[0]["self_attn.q_proj"]["A"]["init"])