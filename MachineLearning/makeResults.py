import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import clone

from tensorflow.keras.models import clone_model

from IPython.display import clear_output
import tqdm

from sklearn.model_selection import StratifiedKFold, LeaveOneOut

import tensorflow as tf
import matplotlib.pyplot as plt
import json


class Results:
    def __init__(self, clf, X, y):
        self.true_labels = y
        self.preds = clf.predict(X)
        self.accuracy = accuracy_score(y, self.preds)
        self.precision, self.recall, self.f1, _ = precision_recall_fscore_support(y, self.preds) 

def get_results(dataset,
                clf):
    
    X_Train, X_Test, y_Train, y_Test = dataset
    clf.fit(X_Train, y_Train)
    
    Test = Results(clf, X_Test, y_Test)
    Train = Results(clf, X_Train, y_Train)
    
    results = {"train" : Train, "test": Test}
    return results


def autolabel(ax, y, barWidth):
    """
    Attach a text label above each bar displaying its height
    """
    y = [np.round(v, 2) for v in y]
    for i, v in enumerate(y):
        offset = min(y)
        ax.text(i - barWidth / 2, v, str(v), color='black', fontsize=8)


def result_bars(fileName, algs):

    # plt.figure(figsize=(8,4), dpi =1000)
    plt.rcParams['font.size'] = 12

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['figure.dpi'] = 1000
    # plt.rcParams["lines.linewidth"] = 1.3

    # monitorColors = {"blue" : '#377eb8',
    #                 "orange" : '#ff7f00'}
    barWidth = 0.8

    saveData = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1 score": {}}

    metrics = ["accuracy", "precision", "recall", "f1"]
    names = ["Accuracy", "Precision", "Recall", "F1 score"]

    metricColors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=1000)
    ax = axs.ravel()
    for i, metric in enumerate(metrics):

        # Get the corresponding metric, except for neural network model
        vals = []
        for key, item in algs.items():
            try:
                val = item.__dict__[metric]
            except:
                val = item[metric]
            vals.append(val)

            saveData[names[i]][key] = val

        if i == 0:
            # print("vals")
            # print(vals)
            vals = [v * 100 for v in vals]
            # print(vals)
            ax[i].set_ylim(0, 100)

        else:
            ax[i].set_ylim(0, 1)
            vals = [np.mean(v) for v in vals]
        # print("")
        # #vals=np.array(vals)
        # print(vals[1])
        # print(algs.keys())
        # print()
        # print(vals)
        # print()
        # print(barWidth)
        # print()
        # print(names[i])
        # print()
        # print(metricColors[i])

        ax[i].bar(algs.keys(),
                  vals,
                  barWidth,
                  label=names[i],
                  color=metricColors[i])

        autolabel(ax[i], vals, barWidth)
        ax[i].set_ylabel(names[i])
        # ax[i].legend()

        if i == 0:
            ax[i].set_yticklabels([str(t) + "%" for t in ax[i].get_yticks()])

        ax[i].set_xticklabels(algs.keys(), rotation=45)
    #print(fileName)
    #jsonName = fileName.strip(".png") + ".json"
    jsonName = "../plots/results/stat/stat50_8.json"

    # fileName = "../plots/results/stat/stat50_8.png"
    # with open(jsonName, 'w') as json_file:
    #     json.dump(saveData, json_file)

    plt.tight_layout()
    plt.savefig(fileName)


# def result_bars_inline(fileName, algs):

#     # plt.figure(figsize=(8,4), dpi =1000)
#     plt.rcParams['font.size'] = 10

#     plt.rcParams['font.family'] = 'serif'
#     plt.rcParams['font.serif'] = 'Times New Roman'
#     # plt.rcParams['figure.dpi'] = 1000
#     # plt.rcParams["lines.linewidth"] = 1.3

#     # monitorColors = {"blue" : '#377eb8',
#     #                 "orange" : '#ff7f00'}
#     barWidth = 0.8

#     saveData = {"Accuracy": {}, "Precision": {}, "Recall": {}, "F1 score": {}}

#     metrics = ["accuracies", "precisions", "recalls", "f1s"]
#     names = ["Accuracy", "Precision", "Recall", "F1 score"]

#     metricColors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
#     fig, axs = plt.subplots(2, 2, figsize=(8, 6), dpi=1000)
#     ax = axs.ravel()
#     for i, metric in enumerate(metrics):

#         # Get the corresponding metric, except for neural network model
#         vals = []
#         for key, item in algs.items():
#             try:
#                 val = item.__dict__[metric]
#             except:
#                 val = item[metric]
#             vals.append(val)

#             saveData[names[i]][key] = val

#         if i == 0:
#             vals = [v * 100 for v in vals]
#             ax[i].set_ylim(0, 100)

#         else:
#             ax[i].set_ylim(0, 1)

#         ax[i].bar(algs.keys(),
#                   vals,
#                   barWidth,
#                   label=names[i],
#                   color=metricColors[i])
#         autolabel(ax[i], vals, barWidth)
#         ax[i].set_ylabel(names[i])
#         # ax[i].legend()

#         if i == 0:
#             ax[i].set_yticklabels([str(t) + "%" for t in ax[i].get_yticks()])

#         ax[i].set_xticklabels(algs.keys(), rotation=45)

#     # with open(fileName, 'w') as json_file:
#     #     json.dump(saveData, json_file)

#     plt.tight_layout()
#     plt.savefig(fileName)


def result_bars_inline(jsonName):

    plt.rcParams['font.size'] = 6
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['figure.dpi'] = 1000
    # plt.rcParams["lines.linewidth"] = 1.3

    # monitorColors = {"blue" : '#377eb8',
    #                 "orange" : '#ff7f00'}
    barWidth = 0.8

    metrics = ["accuracies", "precisions", "recalls", "f1s"]
    names = ["Accuracy", "Precision", "Recall", "F1 score"]

    with open(PATH) as f:
        data = json.load(f)

    metricColors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    metricHatches = ['X', '*', '#', 'o']
    fig, axs = plt.subplots(1, 4, figsize=(7, 2), dpi=1000)
    ax = axs  #.ravel()

    for i, metric in enumerate(list(data.keys())):

        # Get the corresponding metric, except for neural network model
        vals = list(data[metric].values())

        if i == 0:
            vals = [v * 100 for v in vals]
            ax[i].set_ylim(0, 100)

        else:
            ax[i].set_ylim(0, 1)

        ax[i].bar(list(data[metric].keys()), vals)
        ax[i].bar(list(data[metric].keys()),
                  vals,
                  barWidth,
                  label=metric,
                  color=metricColors[i])
        #autolabel(ax[i], vals, barWidth)
        ax[i].set_ylabel(names[i], fontsize=12)
        #ax[i].legend()

        if i == 0:
            ax[i].set_yticklabels([str(t) + "%" for t in ax[i].get_yticks()])

        ax[i].set_xticklabels(list(data[metric].keys()), rotation=45)

    plt.tight_layout()
    savePATH = '/home/redev/Quanta/NIU_covidTest/AllResults/Images/NN_bars/imgs/inline/' + jsonName.split(
        '/')[-1].strip('.json') + '_rfpolynomial.png'
    plt.savefig(savePATH)
    plt.show()


def result_bars_inline_condensed(jsonName):
    plt.rcParams['font.size'] = 6
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'

    barWidth = 0.8

    metrics = ["accuracies", "f1s"]
    names = ["Accuracy", "F1 score"]

    with open(jsonName) as f:
        data = json.load(f)

    metricColors = ["#1b9e77", "#d95f02"]  #, "#7570b3", "#e7298a"]
    metricHatches = ['X', '*', '#', 'o']
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2), dpi=600)
    ax = axs  #.ravel()

    try:
        del data['Precision']
        del data['Recall']
    except:
        print("Couldn't delete precision and recall")

    for i, metric in enumerate(list(data.keys())):

        # Get the corresponding metric, except for neural network model
        vals = list(data[metric].values())

        if i == 0:
            vals = [v * 100 for v in vals]
            ax[i].set_ylim(0, 100)

        else:
            ax[i].set_ylim(0, 1)

        ax[i].bar(list(data[metric].keys()), vals)
        ax[i].bar(list(data[metric].keys()),
                  vals,
                  barWidth,
                  label=metric,
                  color=metricColors[i])
        #autolabel(ax[i], vals, barWidth)
        ax[i].set_ylabel(names[i], fontsize=12)
        #ax[i].legend()

        if i == 0:
            ax[i].set_yticklabels([str(t) + "%" for t in ax[i].get_yticks()])

        ax[i].set_xticklabels(list(data[metric].keys()), rotation=45)

    plt.tight_layout()
    # savePATH = '/home/redev/Quanta/NIU_covidTest/AllResults/Images/NN_bars/imgs/inline/' + jsonName.split(
    #     '/')[-1].strip('.json') + '_condensed' + '.png'
    savePATH = 'AllResults/Images/NN_bars/imgs/inline/' + jsonName.split(
        '/')[-1].strip('.json') + '_condensed' + '.png'

    plt.savefig(savePATH)
    plt.show()


def result_monitor(filePath, history):
    # plt.figure(figsize=(8,4), dpi =1000)
    plt.rcParams['font.size'] = 12

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    # plt.rcParams['figure.dpi'] = 1000
    plt.rcParams["lines.linewidth"] = 1.3

    monitorColors = {"blue": '#377eb8', "orange": '#ff7f00'}

    fig, axs = plt.subplots(1, 2, figsize=(8, 4), dpi=1000)
    # axs[0].set_title("Monitor accuracy")
    axs[0].plot(history.accuracy * 100,
                label="Train",
                color=monitorColors['blue'])
    axs[0].plot(history.val_accuracy * 100,
                label="Validation",
                color=monitorColors['orange'])
    axs[0].set_yticklabels([str(t) + "%" for t in axs[0].get_yticks()])
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xlabel("Epochs")
    axs[0].legend()

    # axs[1].set_title("Monitor loss")
    axs[1].plot(history.loss, label="Train")
    axs[1].plot(history.val_loss, label="Validation")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].legend()

    plt.tight_layout()

    plt.savefig(filePath)
