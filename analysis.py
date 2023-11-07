import re
import os
import glob
import json
import torch
import pickle
import itertools
import statistics
import numpy as np
from datetime import timedelta
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

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

def svd_left(A, B):
    Ua, Sa, Vta = np.linalg.svd(A, full_matrices=False)
    Ub, Sb, Vtb = np.linalg.svd(B, full_matrices=False)

    C = np.diag(Sb) @ Vtb @ Ua @ np.diag(Sa)

    Uc, Sc, Vtc = np.linalg.svd(C, full_matrices=False)

    return Ub @ Uc

def analyze_grassmann(models):
    # Grassmann distance analysis
    grassmann_matrices = rec_dd()
    for model1, model2 in itertools.combinations(list(models.keys()), 2):
        print(model1, model2)
        for layerIndex in [0]:#models[model1].layers:
            for fragment in ["self_attn.q_proj"]:#modles[model1].layers[layerIndex].modules:
                A1 = models[model1].layers[layerIndex].modules[fragment]["A"]["result"].matrix
                B1 = models[model1].layers[layerIndex].modules[fragment]["B"]["result"].matrix
                A2 = models[model2].layers[layerIndex].modules[fragment]["A"]["result"].matrix
                B2 = models[model2].layers[layerIndex].modules[fragment]["B"]["result"].matrix

                Uw1 = svd_left(A1, B1)
                Uw2 = svd_left(A2, B2)

                Ar = Uw1.shape[1]
                Br = Uw2.shape[1]

                if Uw1.shape[0] != Uw2.shape[0]:
                    continue

                grassmann_matrix = np.zeros((Ar, Br))
                for i in range(1, Ar+1):
                    for j in range(1, Br+1):
                        #print(i, j)
                        grassmann_matrix[i-1, j-1] = grassmann(Uw1, Uw2, i, j)

                grassmann_matrices[f"{model1} - {model2}"][layerIndex][fragment]["W"] = grassmann_matrix
                
                for matrix in models[model1].layers[layerIndex].modules[fragment]:
                    AU, _, AVh = np.linalg.svd(models[model1].layers[layerIndex].modules[fragment][matrix]["result"].matrix)
                    BU, _, BVh = np.linalg.svd(models[model2].layers[layerIndex].modules[fragment][matrix]["result"].matrix)

                    if matrix == "A":
                        A = AVh.T
                        B = BVh.T
                        Ar = AU.shape[0]
                        Br = BU.shape[0]
                    elif matrix == "B":
                        A = AU
                        B = BU
                        Ar = AVh.shape[0]
                        Br = BVh.shape[0]

                    if A.shape[0] != B.shape[0]:
                        continue
                    
                    grassmann_matrix = np.zeros((Ar, Br))
                    for i in range(1, Ar+1):
                        for j in range(1, Br+1):
                            #print(i, j)
                            grassmann_matrix[i-1, j-1] = grassmann(A, B, i, j)

                    grassmann_matrices[f"{model1} - {model2}"][layerIndex][fragment][matrix] = grassmann_matrix

    # Save to pickle file
    grassmann_matrices_dict = ddict2dict(grassmann_matrices)
    with open("grassmann/matrices.pickle", "wb") as handle:
        pickle.dump(grassmann_matrices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved Grassmann matrices")
    
    return grassmann_matrices_dict

def split_grassmann(grassmann_matrix):
    upper_diagonal = np.zeros(grassmann_matrix.shape)
    lower_diagonal = np.zeros((min(grassmann_matrix.shape), min(grassmann_matrix.shape)))

    for i in range(grassmann_matrix.shape[0]):
        for j in range(grassmann_matrix.shape[1]):
            if j >= i:
                upper_diagonal[i, j] = grassmann_matrix[i, j]

            if j <= i:
                lower_diagonal[i, j] = grassmann_matrix[i, j]

    return upper_diagonal, lower_diagonal

def plot_grassmann(grassmann_matrices=None):
    if not grassmann_matrices:
        with open("grassmann/matrices.pickle", "rb") as handle:
            grassmann_matrices = pickle.load(handle)

    for comparison in grassmann_matrices:
        print(comparison)
        for layerIndex in grassmann_matrices[comparison]:
            for fragment in grassmann_matrices[comparison][layerIndex]:
                for matrix in grassmann_matrices[comparison][layerIndex][fragment]:
                    grassmann_matrix = grassmann_matrices[comparison][layerIndex][fragment][matrix]
                    upper_diagonal, lower_diagonal = split_grassmann(grassmann_matrix)

                    saveDir = os.path.join("grassmann", "plots", comparison, str(layerIndex), fragment, matrix)
                    if not os.path.isdir(saveDir):
                        os.makedirs(saveDir)

                    fig, ax = plt.subplots()
                    cax = ax.imshow(upper_diagonal, interpolation='nearest', aspect='auto')
                    fig.colorbar(cax)
                    ax.xaxis.tick_bottom()
                    ax.set_title(f"Subspace distance: {comparison}\nlayer {layerIndex}, module {fragment}, matrix {matrix}")
                    ax.set_xlabel("i")
                    ax.set_ylabel("j")
                    plt.savefig(os.path.join(saveDir, "upper.png"))
                    plt.close()

                    fig, ax = plt.subplots()
                    cax = ax.imshow(lower_diagonal, interpolation='nearest', aspect='auto')
                    fig.colorbar(cax)
                    ax.xaxis.tick_bottom()
                    ax.set_title(f"Subspace distance: {comparison}\nlayer {layerIndex}, module {fragment}, matrix {matrix}")
                    ax.set_xlabel("i")
                    ax.set_ylabel("j")
                    plt.savefig(os.path.join(saveDir, "lower.png"))
                    plt.close()

                    #plt.close("all")

    exit()

def analyze_absolute(models):
    # Absolute value change analysis
    absolute_matrices = rec_dd()
    singulars = rec_dd()
    for model in models:
        print(model)
        for layerIndex in models[model].layers:
            for fragment in models[model].layers[layerIndex].modules:
                for matrix in models[model].layers[layerIndex].modules[fragment]:
                    init = models[model].layers[layerIndex].modules[fragment][matrix]["init"].matrix
                    result = models[model].layers[layerIndex].modules[fragment][matrix]["result"].matrix

                    absolute_matrices[model][layerIndex][fragment][matrix] = np.sum(np.absolute(result - init))

                    if layerIndex == 0:
                        singulars[model][layerIndex][fragment][matrix] = np.linalg.svd(result - init)[1]

                    """
                    if model == "alpaca-2-7b-r64" and layerIndex == 0:
                        saveDir = os.path.join("grassmann", "absolutes")
                        if not os.path.isdir(saveDir):
                            os.makedirs(saveDir)

                        fig, ax = plt.subplots()
                        cax = ax.imshow(result - init, interpolation='nearest', aspect='auto')
                        fig.colorbar(cax)
                        ax.xaxis.tick_bottom()
                        plt.savefig(os.path.join(saveDir, f"{model}_{layerIndex}_{fragment}_{matrix}.png"))
                        plt.close()
                    """

    # Save to pickle file
    absolute_matrices_dict = ddict2dict(absolute_matrices)
    with open("grassmann/absolute.pickle", "wb") as handle:
        pickle.dump(absolute_matrices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved absolute matrix differences")

    singulars_dict = ddict2dict(singulars)
    with open("grassmann/singulars.pickle", "wb") as handle:
        pickle.dump(singulars_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved singular values")
    
    return absolute_matrices_dict

def print_absolute(absolute_matrices=None):
    if not absolute_matrices:
        with open("grassmann/absolute.pickle", "rb") as handle:
            absolute_matrices = pickle.load(handle)

    #print(absolute_matrices)

    contextLengths = {7: 4096, 13: 5120, 70: 8192}

    factors = rec_dd() # A * ...
    for model in absolute_matrices:
        for layerIndex in absolute_matrices[model]:
            for fragment in absolute_matrices[model][layerIndex]:
                layerMatch = re.search(r"-([0-9]+)b-r([0-9]+)", model)
                contextLength = contextLengths[int(layerMatch.group(1))]
                r = int(layerMatch.group(2))

                if not factors[contextLength][fragment]:
                    factors[contextLength][fragment]["A"] = []
                    factors[contextLength][fragment]["B"] = []
                factors[contextLength][fragment]["A"].append(r * contextLength / absolute_matrices[model][layerIndex][fragment]["A"])
                factors[contextLength][fragment]["B"].append(r * contextLength / absolute_matrices[model][layerIndex][fragment]["B"])

    for c in factors:
        for f in factors[c]:
            print(c, f, statistics.mean(factors[c][f]["A"]), statistics.stdev(factors[c][f]["A"]), statistics.mean(factors[c][f]["B"]), statistics.stdev(factors[c][f]["B"]))

    exit()

def print_absolute_singular(singulars=None):
    if not singulars:
        with open("grassmann/singulars.pickle", "rb") as handle:
            singulars = pickle.load(handle)

    #print(singulars)

    for model in singulars:
        saveDir = os.path.join("grassmann", "singulars", model)
        if not os.path.isdir(saveDirLinear := os.path.join(saveDir, "linear")):
            os.makedirs(saveDirLinear)
        if not os.path.isdir(saveDirLog := os.path.join(saveDir, "log")):
            os.makedirs(saveDirLog)
        for layer in singulars[model]:
            allFig, allAx = plt.subplots()
            allAx.set_xlabel("Singular value index")
            allAx.set_xlabel("Singular value")
            allAx.set_title(f"Singular values of adapter differences over initialized state:\n{model}, layer {layer}")
            for fragment in singulars[model][layer]:
                for matrix in singulars[model][layer][fragment]:
                    """
                    fig, ax = plt.subplots()
                    cax = ax.imshow(singulars[model][layer][fragment][matrix][np.newaxis, :], interpolation='nearest', aspect='auto')
                    fig.colorbar(cax)
                    ax.xaxis.tick_bottom()
                    plt.savefig(os.path.join(saveDir, f"{layer}_{fragment}_{matrix}.png"))
                    plt.close()
                    """

                    allAx.plot(singulars[model][layer][fragment][matrix], label=f"{fragment}_{matrix}")    

                    fig, ax = plt.subplots()
                    ax.plot(singulars[model][layer][fragment][matrix])
                    ax.set_xlabel("Singular value index")
                    ax.set_xlabel("Singular value")
                    ax.set_title(f"Singular values of adapter differences over initialized state:\n{model}, layer {layer}, {fragment}, {matrix}")
                    plt.savefig(os.path.join(saveDirLinear, f"{layer}_{fragment}_{matrix}.png"))
                    ax.set_yscale("log")
                    plt.savefig(os.path.join(saveDirLog, f"{layer}_{fragment}_{matrix}.png"))
                    plt.close(fig)
            
            allAx.legend()
            plt.savefig(os.path.join(saveDirLinear, f"allAdapters_{layer}.png"))
            allAx.set_yscale("log")
            plt.savefig(os.path.join(saveDirLog, f"allAdapters_{layer}_log.png"))
            plt.close(allFig)
    
    exit()

def plot_loss():
    with open("training/trainer_state/alpaca-2-7b-r64-t.json") as f:
        data = json.load(f)

        train_loss = []
        eval_loss = []
        mmlu_loss = []
        for step in data["log_history"]:
            if "loss" in step:
                train_loss.append((step["step"], step["loss"]))
            elif "eval_loss" in step:
                eval_loss.append((step["step"], step["eval_loss"]))
            elif "mmlu_loss" in step:
                mmlu_loss.append((step["step"], step["mmlu_loss"]))

        plt.plot(*zip(*train_loss), label="Train loss")
        plt.plot(*zip(*eval_loss), label="Eval loss")
        plt.plot(*zip(*mmlu_loss), label="MMLU loss")
        plt.legend()
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Loss over training steps")
        plt.show()

    exit()

def print_runtime():
    runtimes = defaultdict(list)
    costs = {
        7: 0.36,
        13: 0.36,
        70: 1.99
    }

    for filename in os.listdir("training/all_results"):
        with open(os.path.join("training/all_results", filename)) as f:
            modelMatch = re.search(r"-([0-9]+)b-r([0-9]+)", filename)
            paramCount = int(modelMatch.group(1))
            rank = int(modelMatch.group(2))

            if paramCount == 7 and rank == 32:
                continue

            data = json.load(f)
            runtimes[paramCount].append(data["train_runtime"])
            #print(filename, timedelta(seconds=data["train_runtime"]))

    print(runtimes)
    for paramCount in runtimes:
        mean = statistics.mean(runtimes[paramCount])
        print(paramCount, timedelta(seconds=mean), mean / 3600 * costs[paramCount])

    exit()

def print_bleu():
    with open("bleu.pickle", "rb") as f:
        bleuScores = pickle.load(f)

    for paramCount in bleuScores:
        for rank in bleuScores[paramCount]:
            print(paramCount, rank, statistics.mean(bleuScores[paramCount][rank]), statistics.stdev(bleuScores[paramCount][rank]), f"{sum(i < 0.12 for i in bleuScores[paramCount][rank])} / {len(bleuScores[paramCount][rank])}")

    exit()

def analyze(models):
    #grassmann_matrices = analyze_grassmann(models)
    #plot_grassmann(grassmann_matrices)

    absolute_matrices = analyze_absolute(models)
    print_absolute(absolute_matrices)

specificModels = []#["alpaca-2-7b-r64", "alpaca-2-7b-r8"]

if __name__ == '__main__':
    #ensureImageSubset(os.path.join(PATH, "alpaca-2-13b-r64/init-r64-meta-llama/Llama-2-13b-hf/adapter_model.bin"), os.path.join(PATH, "/workspace/analysis/alpaca-2-13b-r32/init-r32-meta-llama/Llama-2-13b-hf/adapter_model.bin"))

    #plot_grassmann()
    #print_absolute()

    #print_runtime()
    #plot_loss()

    # print_absolute_singular()

    print_bleu()

    models = {}
    for directory in os.listdir(PATH):
        if not specificModels or directory in specificModels:
            models[directory] = Model(os.path.join(PATH, directory))

    #print(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape)  # 4096
    #print(models["alpaca-2-13b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape) # 5120
    #print(models["alpaca-2-70b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape) # 8192

    #analyze_absolute(models)

    #plotDistribution(models["alpaca-2-7b-r64"].layers[0]["self_attn.q_proj"]["A"]["init"])

    analyze(models)