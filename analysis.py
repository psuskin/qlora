import re
import os
import glob
import torch
import pickle
import itertools
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

def grassmann(A, B, i, j):
    Ui = A[:, :i]
    Uj = B[:, :j]

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

def analyze_grassmann(models):
    # Grassmann distance analysis
    grassmann_matrices = rec_dd()
    for model1, model2 in itertools.combinations(list(models.keys())):
        for layerIndex in [0]:#model1.layers:
            for fragment in ["self_attn.q_proj"]:#model1.layers[layerIndex].modules:
                for matrix in model1.layers[layerIndex].modules[fragment]:
                    AU, _, AVh = np.linalg.svd(models[model1].layers[layerIndex].modules[fragment][matrix]["result"].matrix)
                    BU, _, BVh = np.linalg.svd(models[model2].layers[layerIndex].modules[fragment][matrix]["result"].matrix)

                    if matrix == "A":
                        A = AVh.T
                        B = BVh.T
                    elif matrix == "B":
                        A = AU
                        B = BU

                    Ar = A.shape[0]
                    Br = B.shape[0]

                    grassmann_matrix = np.zeros((Ar, Br))
                    for i in range(1, Ar+1):
                        for j in range(1, Br+1):
                            grassmann_matrix[i-1, j-1] = grassmann(A, B, i, j)

                    grassmann_matrices[tuple(sorted(model1, model2))][layerIndex][fragment][matrix] = grassmann_matrix

    # Save to pickle file
    with open("grassmann/matrices.pickle", "wb") as handle:
        pickle.dump(grassmann_matrices, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return grassmann_matrices

def split_grassmann(grassmann_matrix):
    upper_diagonal = np.zeros(grassmann_matrix.shape)
    lower_diagonal = np.zeros(grassmann_matrix.shape)

    for i in range(grassmann_matrix.shape[0]):
        for j in range(grassmann_matrix.shape[1]):
            if j >= i:
                upper_diagonal[i, j] = grassmann_matrix[i, j]

            if j <= i:
                lower_diagonal[i, j] = grassmann_matrix[i, j]

    return upper_diagonal, lower_diagonal

def plot_grassmann(grassmann_matrices=None):
    with open("grassmann/matrices.pickle", "rb") as handle:
        grassmann_matrices = pickle.load(handle)

    for model1, model2 in grassmann_matrices:
        for layerIndex in grassmann_matrices[tuple(sorted(model1, model2))]:
            for fragment in grassmann_matrices[tuple(sorted(model1, model2))][layerIndex]:
                for matrix in grassmann_matrices[tuple(sorted(model1, model2))][layerIndex][fragment]:
                    grassmann_matrix = grassmann_matrices[layerIndex][fragment][matrix]
                    upper_diagonal, lower_diagonal = split_grassmann(grassmann_matrix)

                    fig, ax = plt.subplots()

                    ax.matshow(upper_diagonal)
                    plt.savefig(os.path.join("grassmann", "plots", f"{model1} - {model2}", layerIndex, fragment, matrix, "upper.png"))

                    ax.matshow(lower_diagonal)
                    plt.savefig(os.path.join("grassmann", "plots", f"{model1} - {model2}", layerIndex, fragment, matrix, "lower.png"))

def analyze(models):
    grassmann_matrices = analyze_grassmann(models)
    plot_grassmann(grassmann_matrices)

specificModels = []#["alpaca-2-7b-r64", "alpaca-2-7b-r8"]

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

    #plotDistribution(models["alpaca-2-7b-r64"].layers[0]["self_attn.q_proj"]["A"]["init"])

    analyze(models)