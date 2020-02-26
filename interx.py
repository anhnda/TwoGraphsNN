import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger
import numpy as np
from dataFactory.loader2 import BioLoader2


def interx():
    bioLoader2 = BioLoader2()
    trainPath = BioLoader2.getPathIFold(0)
    bioLoader2.createTrainTestGraph(trainPath,allTrain=True)
    from models.MPNNX import MPNNX
    model = MPNNX()
    model.resetModel()
    drugEmbedding, seEmbedding = model.train(bioLoader2, debug=False, pred=False)
    np.savetxt("%s/ALL_DRUG.txt" %config.OUTPUT_DIR, drugEmbedding.detach().numpy())
    np.savetxt("%s/ALL_SE.txt" %config.OUTPUT_DIR, seEmbedding.detach().numpy())

def plotHM():
    from utils.plotHM import plotMatrixHeatMap
    drugEmbedding = np.loadtxt("%s/ALL_DRUG.txt" %config.OUTPUT_DIR)
    seEmbedding = np.loadtxt("%s/ALL_SE.txt" %config.OUTPUT_DIR)

    plotMatrixHeatMap(drugEmbedding, "drugHM")
    plotMatrixHeatMap(seEmbedding, "seHM")

if __name__ == "__main__":
    interx()
    plotHM()

