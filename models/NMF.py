from sklearn.decomposition import NMF
import config
from dataFactory.loader import BioLoader
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def run():
    model = NMF(n_components=config.EMBED_DIM)
    bioLoader = BioLoader()
    trainPath = bioLoader.getPathIFold(1)
    bioLoader.createTrainTestGraph(trainPath)
    out = bioLoader.trainOutMatrix.numpy()
    W = model.fit_transform(out)
    U = model.components_

    re = np.dot(W, U)

    auc = roc_auc_score(out.reshape(-1), re.reshape(-1))
    aupr = average_precision_score(out.reshape(-1), re.reshape(-1))

    print("AUC, AUPR:", auc, aupr)
    print(np.sum(out))


if __name__ == "__main__":
    run()
