import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger
from dataFactory.loader2 import BioLoader2
from dataFactory.loader3 import BioLoader3
from dataFactory.loader4 import BioLoader4

import numpy as np


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

        aucs = []
        auprs = []

        rareAUCs = []
        rareAUPRs = []

        for iFold in range(config.K_FOLD):
            config.IFOLD = iFold

            if model.name == "MPNNX3":
                bioLoader = BioLoader3()
                trainPath = BioLoader3.getPathIFold(iFold)
            elif model.name == "MPNNX4":
                bioLoader = BioLoader4()
                trainPath = BioLoader4.getPathIFold(iFold)
            else:
                bioLoader = BioLoader2()
                trainPath = BioLoader2.getPathIFold(iFold)

            bioLoader.createTrainTestGraph(trainPath)
            logger.infoAll("Training raw path: %s" % trainPath)
            logger.infoAll(("Number of substructures, proteins, pathways, drugs, se: ", config.CHEM_FINGERPRINT_SIZE, bioLoader.nProtein, bioLoader.nPathway, bioLoader.nDrug, bioLoader.nSe))


            inputTrain = bioLoader.trainInpMat
            outputTrain = bioLoader.trainOutMatrix.numpy()
            inputTest = bioLoader.testInpMat
            outputTest = bioLoader.testOutMatrix.numpy()
            seMat = bioLoader.seMat
            if model.name.startswith("MPNN"):
                pred = model.fitAndPredict(bioLoader)
            elif model.name == "CSMF":
                pred = model.fitAndPredict(inputTrain, outputTrain, inputTest, seMat)
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

            rareAUC, rareAUPR = self.rareEval(outputTest, pred, bioLoader.listRareSe)

            logger.infoAll(rareAUC)
            logger.infoAll(rareAUPR)

            rareAUCs.append(rareAUC)
            rareAUPRs.append(rareAUPR)

        meanAuc, seAuc = getMeanSE(aucs)
        meanAupr, seAupr = getMeanSE(auprs)

        logger.infoAll("AUC: %.4f %.4f" % (meanAuc, seAuc))
        logger.infoAll("AUPR: %.4f %.4f" % (meanAupr, seAupr))

        lAllAUC, lAllAUPR = self.mergerRarEval(rareAUCs, rareAUPRs)
        logger.infoAll(("RARE AUC", lAllAUC))
        logger.infoAll(("RARE AUPR", lAllAUPR))




def getAUCAUPR(trueTargets, predicted):
    return roc_auc_score(trueTargets.reshape(-1), predicted.reshape(-1)), \
           average_precision_score(trueTargets.reshape(-1), predicted.reshape(-1))
