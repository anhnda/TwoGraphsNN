import config


def loadRe():
    path = "%s/rePred" % config.OUTPUT_DIR
    fin = open(path)
    modelNames = []
    AUCs = []
    stdAUCs = []
    AUPRs = []
    stdAUPRs = []

    while True:
        line = fin.readline()
        if line == "":
            break
        line = line.strip()
        modelName = line
        modelNames.append(modelName)

        aucString = fin.readline().strip()
        aucParts = aucString.split(" ")
        AUCs.append(float(aucParts[1]))
        stdAUCs.append(float(aucParts[2]))

        auprString = fin.readline().strip()
        auprParts = auprString.split(" ")
        AUPRs.append(float(auprParts[1]))
        stdAUPRs.append(float(auprParts[2]))

    fin.close()

    return modelNames, AUCs, stdAUCs, AUPRs, stdAUPRs


def plot():
    import matplotlib.pyplot as plt
    import numpy as np
    modelNames, AUCs, stdAUCs, AUPRs, stdAUPRs = loadRe()
    res = [[AUCs, stdAUCs], [AUPRs, stdAUPRs]]
    METRICS = ["AUC", "AUPR"]

    SELECTED_MODELS = ["2Graph", "NeuSK", "CSMF", "SCCA"]
    SELECTED_INDICES = []
    dModelName2Index = dict()
    for idx, modelName in enumerate(modelNames):
        dModelName2Index[modelName] = idx

    for selectedModel in SELECTED_MODELS:
        SELECTED_INDICES.append(dModelName2Index[selectedModel])

    width = 0.5
    xInd = np.arange(len(SELECTED_MODELS))

    def getSubList(ls, ids):
        subList = []
        for idx in ids:
            subList.append(ls[idx])
        return subList

    for i in range(2):
        fig, ax = plt.subplots()
        # PLOTS = tuple()
        METRIC = METRICS[i]

        aucs, stdAUCs = res[i]
        sAucs = getSubList(aucs, SELECTED_INDICES)
        sstdAUCs = getSubList(stdAUCs, SELECTED_INDICES)

        p1 = ax.bar(xInd, sAucs, width, bottom=0, yerr=sstdAUCs)
        # PLOTS += (p1[0],)
        # ax.set_title(re['SE_SCHEMAS'][SE_SCHEMAS_ID])
        ax.set_ylabel('%s' % METRIC)
        ax.set_xlabel('Models')

        ax.set_xticks(xInd)
        ax.set_xticklabels(SELECTED_MODELS)
        ax.tick_params(axis='x', length=0, width=0)
        # ax.legend(PLOTS,re['MODELS'])

        if METRIC == 'AUC':
            ax.set_ylim(bottom=0.7)
        else:
            ax.set_ylim(bottom=0.4)

        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0 + box.height * 0.1,
        #                 box.width, box.height * 0.9])
        # fig.legend(PLOTS, SELECTED_MODELS, loc='upper center', ncol=len(SELECTED_MODELS), fancybox=True,
        #           bbox_to_anchor=(0.54, 0.95), handlelength=1.4)

        plt.savefig("%s/%s_Pred.png" % (config.FIG_DIR, METRIC))
        plt.savefig("%s/%s_Pred.eps" % (config.FIG_DIR, METRIC))


if __name__ == "__main__":
    plot()
