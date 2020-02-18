import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger
from dataFactory.loader2 import BioLoader2

import numpy as np


def getMeanSE(ar):
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se


class Evaluator:

    def __init__(self):
        pass

    def evalModel(self, model):
        # Logger
        from utils import utils
        utils.ensure_dir("%s/logs" % config.C_DIR)
        logPath = "%s/logs/%s_%s" % (config.C_DIR, model.name, utils.getCurrentTimeString())
        logger = MyLogger(logPath)

        logger.infoAll(model.getParams())
        logger.infoAll(model)

        aucs = []
        auprs = []
        for iFold in range(config.K_FOLD):
            config.IFOLD = iFold
            bioLoader2 = BioLoader2()
            trainPath = BioLoader2.getPathIFold(iFold)
            bioLoader2.createTrainTestGraph(trainPath)

            logger.infoAll("Training raw path: %s" % trainPath)

            inputTrain = bioLoader2.trainInpMat
            outputTrain = bioLoader2.trainOutMatrix.numpy()
            inputTest = bioLoader2.testInpMat
            outputTest = bioLoader2.testOutMatrix.numpy()

            if model.name == "MPNNX":
                pred = model.fitAndPredict(bioLoader2)
            else:
                pred = model.fitAndPredict(inputTrain, outputTrain, inputTest)

            if model.repred is None:
                repred = model.predict(inputTrain)
            else:
                repred = model.repred
            # print("Shape: ", outputTrain.shape, repred.shape, type(outputTrain), type(repred))
            auctrain = roc_auc_score(outputTrain.reshape(-1), repred.reshape(-1))
            auprtrain = average_precision_score(outputTrain.reshape(-1), repred.reshape(-1))

            auc = roc_auc_score(outputTest.reshape(-1), pred.reshape(-1))
            aupr = average_precision_score(outputTest.reshape(-1), pred.reshape(-1))
            logger.infoAll("Train: %.4f %.4f" % (auctrain, auprtrain))
            logger.infoAll("Test: %.4f %.4f" % (auc, aupr))
            aucs.append(auc)
            auprs.append(aupr)
            print(iFold)

        meanAuc, seAuc = getMeanSE(aucs)
        meanAupr, seAupr = getMeanSE(auprs)

        logger.infoAll("AUC: %.4f %.4f" % (meanAuc, seAuc))
        logger.infoAll("AUPR: %.4f %.4f" % (meanAupr, seAupr))


def getAUCAUPR(trueTargets, predicted):
    return roc_auc_score(trueTargets.reshape(-1), predicted.reshape(-1)), \
           average_precision_score(trueTargets.reshape(-1), predicted.reshape(-1))
