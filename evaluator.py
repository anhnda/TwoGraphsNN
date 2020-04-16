import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger

from dataFactory.loader import BioLoader5P2

import numpy as np
import random
import torch


def getMeanSE(ar):
    mean = np.mean(ar)
    se = np.std(ar) / np.sqrt(len(ar))
    return mean, se


class Evaluator:

    def __init__(self):
        pass

    def rareEval(self, target, output, listIds):
        aucList = list()
        auprList = list()
        for l in listIds:
            if len(l) == 0:
                continue
            targetx = target[:, l]
            outputx = output[:, l]
            auc = roc_auc_score(targetx.reshape(-1), outputx.reshape(-1))
            aupr = average_precision_score(targetx.reshape(-1), outputx.reshape(-1))
            aucList.append(auc)
            auprList.append(aupr)
        return aucList, auprList

    def mergerRarEval(self, aucs, auprs):
        nEval = len(aucs[0])
        nFold = len(aucs)
        lAllAUC = list()
        lAllAUPR = list()
        for j in range(nEval):
            lauc = list()
            laupr = list()
            for i in range(nFold):
                lauc.append(aucs[i][j])
                laupr.append(auprs[i][j])
            lAllAUC.append(getMeanSE(lauc))
            lAllAUPR.append(getMeanSE(laupr))

        return lAllAUC, lAllAUPR

    def evalModel(self, model):
        # Logger
        from utils import utils
        utils.ensure_dir("%s/logs" % config.C_DIR)
        logPath = "%s/logs/%s_%s" % (config.C_DIR, model.name, utils.getCurrentTimeString())
        logger = MyLogger(logPath)

        logger.infoAll(model.getParams())
        logger.infoAll(model)
        # logger.infoAll(("Torch seed: ", torch.seed()))
        logger.infoAll(("Manual torch seed: ", config.TORCH_SEED))
        logger.infoAll(("KFold: %s x %s" % (config.NTIMES_KFOLD, config.K_FOLD)))
        logger.infoAll(("Optimizer: ", config.OPTIMIZER))
        torch.manual_seed(config.TORCH_SEED)
        np.random.seed(config.TORCH_SEED)
        random.seed(config.TORCH_SEED)

        model.setLogger(logger)

        aucs = []
        auprs = []

        for timeFold in range(config.NTIMES_KFOLD):
            for iFold in range(config.K_FOLD):
                config.IFOLD = iFold

                bioLoader = BioLoader5P2()
                trainPath = BioLoader5P2.getPathNTIMESIFold(timeFold, iFold)

                bioLoader.createTrainTestVal(trainPath)
                logger.infoAll("Training raw path: %s" % trainPath)
                logger.infoAll(
                    ("Number of substructures, proteins, pathways, drugs, se: ", config.CHEM_FINGERPRINT_SIZE,
                     bioLoader.nProtein, bioLoader.nPathway, bioLoader.nDrug, bioLoader.nSe))

                inputTrain = bioLoader.trainInpMat
                inputTest = bioLoader.testInpMat

                targetTrain = bioLoader.trainOutMatrix.numpy()
                targetTest = bioLoader.testOutMatrix.numpy()

                logger.infoAll((inputTrain.shape, inputTest.shape, targetTrain.shape, targetTest.shape))



                print(inputTrain.shape, targetTrain.shape)
                testOut = model.fitAndPredict(bioLoader)
                trainOut = model.repred
                # print("Shape: ", outputTrain.shape, repred.shape, type(outputTrain), type(repred))
                auctrain = roc_auc_score(targetTrain.reshape(-1), trainOut.reshape(-1))
                auprtrain = average_precision_score(targetTrain.reshape(-1), trainOut.reshape(-1))

                auc = roc_auc_score(targetTest.reshape(-1), testOut.reshape(-1))
                aupr = average_precision_score(targetTest.reshape(-1), testOut.reshape(-1))
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
