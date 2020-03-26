import config
import ast
def loadData():
    path = "%s/rareSE" % config.OUTPUT_DIR
    modelNames = []
    reAUCs = []
    reAUPRs = []
    fin = open(path)


    parts = fin.readline().strip().split(" ")
    nDrug = int(parts[0])
    offsetSe = float(parts[1])
    stepFloat = float(parts[2])
    stepSize = int(stepFloat * nDrug)
    offsetSize = int(nDrug * offsetSe)
    X = []
    plotSize = 0.05
    plotSkip = int(plotSize / stepFloat)

    isUpdateX = True
    while True:
        line = fin.readline()
        if line == "":
            break

        modelName = line.strip()
        modelNames.append(modelName)

        reAUC = fin.readline().strip()
        reAUC = ast.literal_eval(reAUC)
        aucs  = list()
        errs = list()

        ix = -1
        for p in reAUC[1]:
            ix += 1
            if ix % plotSkip == 0:
                if isUpdateX:
                    X.append(offsetSize + stepSize * ix - 1)
                a, e = p[0], p[1]
                aucs.append(a)
                errs.append(e)
        isUpdateX = False

        reAUCs.append([aucs, errs])

        reAUPR = fin.readline().strip()
        reAUPR = ast.literal_eval(reAUPR)
        auprs = list()
        err2s = list()
        ix = -1
        for p in reAUPR[1]:
            ix += 1
            if ix % plotSkip == 0:
                a, e = p
                auprs.append(a)
                err2s.append(e)

        reAUPRs.append([auprs, err2s])
    fin.close()

    return modelNames, reAUCs, reAUPRs, X


def plotErrBar():
    modelNames, reAUCs, reAUPRs, X = loadData()
    import matplotlib.pyplot as plt


    TYPES = ['AUC', 'AUPR']
    RES = [reAUCs, reAUPRs]
    LINE_STYLES = ['-', '--', ':' ]

    for j in range(2):
        typex = TYPES[j]
        rex = RES[j]
        fig = plt.figure()

        for i in range(len(modelNames)):
            label = modelNames[i]
            v, errs = rex[i]
            plt.errorbar(X, v, yerr=errs, label= label, linestyle=LINE_STYLES[i])
        plt.xlabel('Num Drugs/ SE')
        plt.ylabel(typex)

        plt.legend(loc='lower right')
        plt.tight_layout()

        plt.savefig("%s/%s.eps" % (config.FIG_DIR, typex))
        plt.savefig("%s/%s.png" % (config.FIG_DIR, typex))



if __name__ == "__main__":
    # print (loadData())
    plotErrBar()



