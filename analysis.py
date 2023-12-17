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

PATH = "/home/psuskin/repos/analysis"

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

                    if comparison == "alpaca-2-7b-r8 - alpaca-2-7b-r64" and matrix == "W":
                        print("Upper diagonal", np.min(upper_diagonal[upper_diagonal != 0]), np.max(upper_diagonal))
                        for i in range(upper_diagonal.shape[0]):
                            for j in range(upper_diagonal.shape[1]):
                                if upper_diagonal[i, j] != 0:
                                    print(j+1, 8-i, upper_diagonal[i, j])

                        print("Lower diagonal")
                        for i in range(lower_diagonal.shape[0]):
                            for j in range(lower_diagonal.shape[1]):
                                if lower_diagonal[i, j] != 0:
                                    print(j+1, 8-i, lower_diagonal[i, j])
                    else:
                        continue

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
                    #plt.savefig(os.path.join(saveDir, "upper.png"))
                    plt.close()

                    fig, ax = plt.subplots()
                    cax = ax.imshow(lower_diagonal, interpolation='nearest', aspect='auto')
                    fig.colorbar(cax)
                    ax.xaxis.tick_bottom()
                    ax.set_title(f"Subspace distance: {comparison}\nlayer {layerIndex}, module {fragment}, matrix {matrix}")
                    ax.set_xlabel("i")
                    ax.set_ylabel("j")
                    #plt.savefig(os.path.join(saveDir, "lower.png"))
                    plt.close()

                    #plt.close("all")

    exit()

def analyze_absolute(models):
    # Absolute value change analysis
    absolute_matrices = rec_dd()
    singulars = rec_dd()
    differences = rec_dd()
    for model in models:
        print(model)
        for layerIndex in models[model].layers:
            for fragment in models[model].layers[layerIndex].modules:
                for matrix in models[model].layers[layerIndex].modules[fragment]:
                    init = models[model].layers[layerIndex].modules[fragment][matrix]["init"].matrix
                    result = models[model].layers[layerIndex].modules[fragment][matrix]["result"].matrix

                    absolute_matrices[model][layerIndex][fragment][matrix] = np.sum(np.absolute(result - init))

                    singulars[model][layerIndex][fragment][matrix] = np.linalg.svd(result - init, compute_uv=False)

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
                    if model == "alpaca-2-7b-r64" and layerIndex == 0:
                        differences[model][layerIndex][fragment][matrix] = result - init

    # Save to pickle file
    absolute_matrices_dict = ddict2dict(absolute_matrices)
    with open("grassmann/absolute.pickle", "wb") as handle:
        pickle.dump(absolute_matrices_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved absolute matrix differences")

    singulars_dict = ddict2dict(singulars)
    with open("grassmann/singulars.pickle", "wb") as handle:
        pickle.dump(singulars_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved singular values")

    differences_dict = ddict2dict(differences)
    with open("grassmann/differences.pickle", "wb") as handle:
        pickle.dump(differences_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved differences")
    
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
                factors[contextLength][fragment]["A"].append(absolute_matrices[model][layerIndex][fragment]["A"] / (r * contextLength))
                factors[contextLength][fragment]["B"].append(absolute_matrices[model][layerIndex][fragment]["B"] / (r * contextLength))

    for c in factors:
        for f in factors[c]:
            print(c, f, 1 / statistics.mean(factors[c][f]["A"]), 1 / statistics.stdev(factors[c][f]["A"]), 1 / statistics.mean(factors[c][f]["B"]), 1 / statistics.stdev(factors[c][f]["B"]))

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
                    #plt.savefig(os.path.join(saveDirLinear, f"{layer}_{fragment}_{matrix}.png"))
                    ax.set_yscale("log")
                    #plt.savefig(os.path.join(saveDirLog, f"{layer}_{fragment}_{matrix}.png"))
                    plt.close(fig)

                    # if model == "alpaca-2-7b-r64" and layer == 0:
                    #     print(fragment, matrix)
                    #     print(*zip(range(1, len(singulars[model][layer][fragment][matrix])+1), singulars[model][layer][fragment][matrix]))

                    # if model == "alpaca-2-7b-r8" and layer == 0:
                    #     print(fragment, matrix)
                    #     print(*zip(range(1, len(singulars[model][layer][fragment][matrix])+1), singulars[model][layer][fragment][matrix]))

                    if model == "alpaca-2-7b-r64" and layer == 0:
                        print(fragment, matrix)
                        print(*zip(range(1, 9), singulars[model][layer][fragment][matrix][:8]))
            
            allAx.legend()
            #plt.savefig(os.path.join(saveDirLinear, f"allAdapters_{layer}.png"))
            allAx.set_yscale("log")
            #plt.savefig(os.path.join(saveDirLog, f"allAdapters_{layer}_log.png"))
            plt.close(allFig)
    
    exit()

def plot_loss():
    with open("training/trainer_state/alpaca-2-7b-r64-t.json") as f:
        data = json.load(f)

        train_loss = []
        eval_loss = [(1, 3.271456241607666)]
        mmlu_loss = [(1, 2.9779934809147184)]
        for step in data["log_history"]:
            if "loss" in step:
                train_loss.append((step["step"], step["loss"]))
            elif "eval_loss" in step:
                eval_loss.append((step["step"], step["eval_loss"]))
            elif "mmlu_loss" in step:
                mmlu_loss.append((step["step"], step["mmlu_loss"]))

        print(*train_loss)
        print(*eval_loss)
        print(*mmlu_loss)

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

def max_index(lst, single=False):
    if not lst:
        return -1

    max_val = max(lst)
    max_indices = [i for i, val in enumerate(lst) if val == max_val]

    if single:
        if len(max_indices) == 1:
            return max_indices[0]
        else:
            return -1
    else:
        return max_indices

def print_bleu():
    with open("bleu.pickle", "rb") as f:
        bleuScoresOrig = pickle.load(f)

    with open("data/en_articles_alpaca.json", encoding="utf-8") as f:
        data = json.load(f)
    with open("evalSamples.json", encoding="utf-8") as f:
        evalSamples = json.load(f)
    evalIndices = []
    for sample in evalSamples:
        if (index := data.index(sample)) % 2:
            evalIndices.append((index - 1) // 2)
    print(np.asarray(bleuScoresOrig[7][64])[evalIndices])

    bleuScoresNoEval = rec_dd()
    for paramCount in bleuScoresOrig:
        for rank in sorted(bleuScoresOrig[paramCount]):
            plt.plot(bleuScoresOrig[paramCount][rank], label=f"{paramCount}-{rank}")
            bleuScoresNoEval[paramCount][rank] = [score for i, score in enumerate(bleuScoresOrig[paramCount][rank]) if i not in evalIndices]
    #plt.show()
    plt.close()

    for bleuScores in [bleuScoresOrig, bleuScoresNoEval]:
        colors = {
            0: "gray",
            64: "blue",
            32: "orange",
            16: "green",
            8: "red",
            4: "purple",
            2: "brown",
            1: "pink"
        }
        for paramCount in bleuScores:
            for rank in sorted(bleuScores[paramCount]):
                values, bins = np.histogram(bleuScores[paramCount][rank], bins=100)
                #print(bins)

                cumulative = np.cumsum(values)
                plt.plot(bins[:-1], cumulative, label=f"{paramCount}-{rank}: {sum(i == 1 for i in bleuScores[paramCount][rank])} perfect scores", linestyle="dashed" if paramCount == 13 else None, color=colors[rank])

                print(paramCount, rank)
                print(*zip(bins[:-1], cumulative))

                # plt.plot(bins[:-2], values[:-1], label=f"{paramCount}-{rank}")

                print(paramCount, rank, statistics.mean(bleuScores[paramCount][rank]), statistics.stdev(bleuScores[paramCount][rank]), f"{sum(i < 0.12 for i in bleuScores[paramCount][rank])} / {len(bleuScores[paramCount][rank])}", f"\t{sum(i == 1 for i in bleuScores[paramCount][rank])} / {len(bleuScores[paramCount][rank])}")
        plt.legend()
        plt.xlabel("BLEU score")
        plt.ylabel("Cumulative count (dataset size of 630 samples)")
        plt.show()

        models = [f"{paramCount}-{rank}" for paramCount in bleuScores for rank in bleuScores[paramCount]]
        singleWinners = defaultdict(int)
        multiWinners = defaultdict(int)
        for scores in zip(*[bleuScores[paramCount][rank] for paramCount in bleuScores for rank in bleuScores[paramCount]]):
            if (winner := max_index(scores, single=True)) != -1:
                singleWinners[models[winner]] += 1
            for winner in max_index(scores):
                multiWinners[models[winner]] += 1
        print(singleWinners)
        print(multiWinners)

    exit()

def print_differences(differences=None):
    if not differences:
        with open("grassmann/differences.pickle", "rb") as handle:
            differences = pickle.load(handle)

    difference = differences["alpaca-2-7b-r64"][0]["self_attn.q_proj"]["A"]

    fig, ax = plt.subplots()
    cax = ax.imshow(difference, interpolation='nearest', aspect='auto')
    fig.colorbar(cax)
    ax.xaxis.tick_bottom()
    plt.savefig("difference.pdf")

    exit()

def analyze(models):
    #grassmann_matrices = analyze_grassmann(models)
    #plot_grassmann(grassmann_matrices)

    absolute_matrices = analyze_absolute(models)
    print_absolute(absolute_matrices)

specificModels = []#["alpaca-2-7b-r64", "alpaca-2-7b-r8"]

import scipy.stats as stats
def plotNF4():
    values = [-1.0, -0.6961928009986877, -0.5250730514526367,
-0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
-0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
0.24611230194568634, 0.33791524171829224, 0.44070982933044434,
0.5626170039176941, 0.7229568362236023, 1.0]

    mu = 0
    sigma = 1 / 1.848
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    for value in values:
        # plt.axvline(x=value, color="red")
        #print(f"({value}, 0)\n({value}, {stats.norm.pdf(value, mu, sigma)})")
        plt.vlines(x=value, ymin=0, ymax=stats.norm.pdf(value, mu, sigma), color="red")
    plt.show()

    exit()

if __name__ == '__main__':
    #plotNF4()

    #ensureImageSubset(os.path.join(PATH, "alpaca-2-13b-r64/init-r64-meta-llama/Llama-2-13b-hf/adapter_model.bin"), os.path.join(PATH, "/workspace/analysis/alpaca-2-13b-r32/init-r32-meta-llama/Llama-2-13b-hf/adapter_model.bin"))

    #plot_grassmann()
    print_absolute()

    #print_runtime()
    #plot_loss()

    #print_absolute_singular()
    #print_differences()

    #print_bleu()

    models = {}
    for directory in os.listdir(PATH):
        if not specificModels or directory in specificModels:
            models[directory] = Model(os.path.join(PATH, directory))

    #print(models["alpaca-2-7b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape)  # 4096
    #print(models["alpaca-2-13b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape) # 5120
    #print(models["alpaca-2-70b-r64"].layers[0].modules["self_attn.q_proj"]["A"]["init"].matrix.shape) # 8192

    #analyze_absolute(models)

    #plotDistribution(models["alpaca-2-7b-r64"].layers[0]["self_attn.q_proj"]["A"]["init"])

    #analyze(models)