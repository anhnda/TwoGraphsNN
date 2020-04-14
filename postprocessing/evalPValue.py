from scipy import stats


def loadLogFile(path):
    auc = []
    aupr = []
    fin = open(path)
    xauc = ""
    xaupr = ""
    while True:
        line = fin.readline()
        if line == "":
            break
        if line.startswith("Test:"):
            parts = line.strip().split(" ")
            auc.append(float(parts[1]))
            aupr.append(float(parts[2]))
        elif line.startswith("AUC:"):
            line = line.strip().split(" ")
            xauc = "$%s \pm %s$" % (line[1], line[2])
        elif line.startswith("AUPR:"):
            line = line.strip().split(" ")
            xaupr = "$%s \pm %s$" % (line[1], line[2])
    fin.close()

    return auc, aupr, xauc, xaupr


def compare(path1, path2):
    import numpy as np

    auc1, aupr1, xauc1, xaupr1 = loadLogFile(path1)
    auc2, aupr2, xauc2, xaupr2 = loadLogFile(path2)
    # print (len(auc1), len(auc2))
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

    print("tAUPR: ", tAUPR, "tAUC: ", tAUC)

    print(xaupr2)
    print(xauc2)
    print
    # print("wAUC ", wAUC, " wAUPR", wAUPR)

    return tAUC, tAUPR, wAUC, wAUPR


def run():
    import config

    methods = ["G3N", "SAGE2", "GAT2", "GCNConv2", "SAGE1", "GAT1", "GCNConv1"]
    # methods = ["NESTEDG0.01", "NESTEDG0.05", "NESTEDG0.10", "NESTEDG0.50", "NESTEDG1.00", "NESTEDG0.00"]
    pathMethod0 = "%s/logs/%s" % (config.C_DIR, methods[0])
    for i in range(len(methods)):
        pathMethodi = "%s/logs/%s" % (config.C_DIR, methods[i])
        print (methods[0], methods[i])
        compare(pathMethod0, pathMethodi)


if __name__ == "__main__":
    run()
