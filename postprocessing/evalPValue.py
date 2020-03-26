from scipy import stats


def loadLogFile(path):
    auc = []
    aupr = []
    fin = open(path)
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith("Test:"):
            parts = line.strip().split(" ")
            auc.append(float(parts[1]))
            aupr.append(float(parts[2]))
    fin.close()

    return auc, aupr


def compare(path1, path2):
    import numpy as np

    auc1, aupr1 = loadLogFile(path1)
    auc2, aupr2 = loadLogFile(path2)
    tAUC = stats.ttest_rel(auc1, auc2).pvalue/2
    # tAUC = stats.ttest_ind(auc1, auc2)

    tAUPR = stats.ttest_rel(aupr1, aupr2).pvalue/2
    # tAUPR = stats.ttest_ind(aupr1, aupr2)

    if path1 != path2:
        wAUC = stats.wilcoxon(auc1, auc2, alternative="greater").pvalue
        wAUPR = stats.wilcoxon(aupr1, aupr2, alternative="greater").pvalue
    else:
        wAUC = None
        wAUPR = None

    # print(np.mean(auc1), auc1)
    # print(np.mean(auc2), auc2)
    #
    # print(np.mean(aupr1), aupr1)
    # print(np.mean(aupr2), aupr2)

    print("tAUC: ", tAUC, " tAUPR: ", tAUPR)
    print("wAUC ", wAUC, " wAUPR", wAUPR)

    return tAUC, tAUPR, wAUC, wAUPR


def run():
    import config

    methods = ["NestedInter0", "NestedInter1", "NestedInter2", "NestedOnly", "Outer+InnerFeature", "Outer", "Inner",
               "NeuSK_Inner", "NeuSK_Outer",
               "NeuSK_Both"]
    pathMethod0 = "%s/logs/%s" % (config.C_DIR, methods[0])
    for i in range(len(methods)):
        pathMethodi = "%s/logs/%s" % (config.C_DIR, methods[i])
        print (methods[0], methods[i])
        compare(pathMethod0, pathMethodi)


if __name__ == "__main__":
    run()
