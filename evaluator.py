import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger
from dataFactory.loader2 import BioLoader2
from dataFactory.loader3 import BioLoader3
from dataFactory.loader4 import BioLoader4
from dataFactory.loader4P2 import BioLoader4P2
from dataFactory.loader5P2 import BioLoader5P2

from dataFactory.loader5 import BioLoader5
from dataFactory.loader6 import BioLoader6

from dataFactory.loader1_3 import BioLoader1
from dataFactory.loader3_2 import BioLoader3_2
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
        logger.infoAll(("Config: ", "Protein Test: ", config.PROTEIN_TEST, "KNN: ", config.CS_KNN))
        logger.infoAll(("KFold: %s x %s" % (config.NTIMES_KFOLD, config.K_FOLD)))
        torch.manual_seed(config.TORCH_SEED)
        np.random.seed(config.TORCH_SEED)
        random.seed(config.TORCH_SEED)

        aucs = []
        auprs = []

        rareAUCs = []
        rareAUPRs = []
        for timeFold in range(config.NTIMES_KFOLD):
            for iFold in range(config.K_FOLD):
                config.IFOLD = iFold

                # if model.name == "MPNNX3":
                #     bioLoader = BioLoader3()
                #     trainPath = BioLoader3.getPathIFold(iFold)
                # elif model.name == "MPNNX32":
                #     bioLoader = BioLoader3_2()
                #     trainPath = BioLoader3_2.getPathIFold(iFold)
                # elif model.name == "MPNN1":
                #     bioLoader = BioLoader1()
                #     trainPath = BioLoader1.getPathIFold(iFold)
                # elif model.name.startswith("MPNNX4"):
                #     bioLoader = BioLoader4()
                #     trainPath = BioLoader4.getPathIFold(iFold)
                # elif model.name.startswith("MPNNXP4"):
                #     bioLoader = BioLoader4P2()
                #     trainPath = BioLoader4P2.getPathIFold(iFold)
                if model.name.startswith("MPNNXP5"):
                    bioLoader = BioLoader5P2()
                    trainPath = BioLoader5P2.getPathNTIMESIFold(timeFold, iFold)
                # elif model.name.startswith("MPNNX5"):
                #     bioLoader = BioLoader5()
                #     trainPath = BioLoader5.getPathIFold(iFold)
                # elif model.name.startswith("MPNNX6"):
                #     bioLoader = BioLoader6()
                #     trainPath = BioLoader6.getPathIFold(iFold)

                else:
                    bioLoader = BioLoader2()
                    trainPath = BioLoader2.getPathNTimeIFold(timeFold, iFold)

                bioLoader.createTrainTestGraph(trainPath)
                logger.infoAll("Training raw path: %s" % trainPath)
                logger.infoAll(("Number of substructures, proteins, pathways, drugs, se: ", config.CHEM_FINGERPRINT_SIZE,
                                bioLoader.nProtein, bioLoader.nPathway, bioLoader.nDrug, bioLoader.nSe))

                # inputTrain = bioLoader.trainInpMat[:,config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader.nProtein]
                # inputTest = bioLoader.testInpMat[:,config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader.nProtein]

                inputTrain = bioLoader.trainInpMat
                inputTest = bioLoader.testInpMat

                if not config.PROTEIN_TEST:
                    inputTest[:, config.CHEM_FINGERPRINT_SIZE:].fill(0)
                if config.ONLY_CHEM_FEATURE:
                    inputTrain[:, config.CHEM_FINGERPRINT_SIZE:].fill(0)
                    inputTest[:, config.CHEM_FINGERPRINT_SIZE:].fill(0)
                elif config.ONLY_PROTEIN_FEATURE:
                    inputTrain[:, :config.CHEM_FINGERPRINT_SIZE].fill(0)
                    inputTest[:, :config.CHEM_FINGERPRINT_SIZE:].fill(0)


                # inputTrain = bioLoader.trainInpMat[:, config.CHEM_FINGERPRINT_SIZE + bioLoader.nProtein: ]
                # inputTest = bioLoader.testInpMat[:, config.CHEM_FINGERPRINT_SIZE + bioLoader.nProtein: ]

                outputTrain = bioLoader.trainOutMatrix.numpy()
                outputTest = bioLoader.testOutMatrix.numpy()

                logger.infoAll((inputTrain.shape, inputTest.shape, outputTrain.shape, outputTest.shape))

                logger.infoAll(("VALIDATE SUM PROTEIN TRAIN: ",np.sum(inputTrain[:, config.CHEM_FINGERPRINT_SIZE:])))
                logger.infoAll(("VALIDATE SUM PROTEIN TEST: ",np.sum(inputTest[:, config.CHEM_FINGERPRINT_SIZE:])))

                seMat = bioLoader.seMat

                print(inputTrain.shape, outputTrain.shape)
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

                pred = np.nan_to_num(pred)
                repred = np.nan_to_num(repred)
                outputTest = np.nan_to_num(outputTest)
                outputTrain = np.nan_to_num(outputTrain)

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
