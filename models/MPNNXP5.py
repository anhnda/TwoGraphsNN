from models.net5_2 import Net52
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import numpy as np

from utils import utils
from utils.logger.logger2 import MyLogger
import inspect


class MPNNXP5:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logPath = "%s/logs/MPNNXP52_%s" % (config.C_DIR, utils.getCurrentTimeString())
        print("Logging path: ", logPath)
        self.logger = MyLogger(logPath)

        self.logger.infoAll(inspect.getsource(Net52))

        self.logger.infoAll(("Undirected graph: ", config.UN_DIRECTED))
        self.logger.infoAll(("Protein Test: ", config.PROTEIN_TEST))
        self.logger.infoAll(("Inner Graph, Outer Graph, Se Graph: ",
                             config.INNER_GRAPH, config.OUTER_GRAPH, config.SE_GRAPH))
        self.logger.infoAll(("Drug Features: ", config.INNER_FEATURE))
        self.logger.infoAll(("Combine Features: ", config.COMBINE_FEATURE))

        self.logger.infoAll(("Inner mode: ", config.EXT_MODE))
        self.logger.infoAll(("Cross Prob: ", config.CROSS_PROB))
        self.logger.infoAll(("Inner Level: ", config.INTER_LEVELS))
        self.name = "MPNNXP52"
        self.isFitAndPredict = True

    def resetModel(self, numAtomFeature):
        self.net = Net52(numNode=config.MAX_NODE + 1, numAtomFeature=numAtomFeature)
        self.model = self.net.to(self.device)

    def __getF2Err(self, err):
        err = torch.mul(err, err)
        err = torch.sum(err)
        # err =  self.mseLoss(err,self.zeroTargets)
        return err

    def __getF1Err(self, err):
        err = torch.abs(err)
        err = torch.sum(err)
        # err =  self.mseLoss(err,self.zeroTargets)
        return err

    def getMatDotLoss(self, u, v, t):

        loss = 0
        out = self.model.cal(u, v)

        err1 = out - t
        err1 = self.__getF2Err(err1)
        loss += err1

        # reg1 = self.__getF1Err(u) + self.__getF1Err(v)
        # reg1 *= 0.01
        # loss += reg1

        return loss, out

    def getMatDotLoss2(self, u, v, t):

        loss = 0
        out = torch.matmul(u, v.t())
        # out = torch.sigmoid(out)
        err1 = out - t
        err1 = self.__getF2Err(err1)
        loss += err1

        # reg1 = self.__getF1Err(u) + self.__getF1Err(v)
        # reg1 *= 0.01
        # loss += reg1

        return loss, out

    def train(self, bioLoader5P, debug=True, pred=True):
        optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)
        loss = torch.nn.MSELoss()
        # loss = torch.nn.BCELoss()

        # proteinMat = bioLoader5P.matDrugProtein
        # proteinWeight = np.sum(proteinMat, axis=0)
        # proteinWeight /= np.max(proteinWeight)

        allOuts = []
        allEval = []
        allValErros = []
        allTestErros = []

        target2 = bioLoader5P.trainOutMatrix.numpy()
        validateOutput = bioLoader5P.validOutMatrix.numpy()
        targetTest = bioLoader5P.testOutMatrix.numpy()

        for epoch in range(config.N_EPOCH):

            print("\r%s" % epoch, end="")
            optimizer.zero_grad()
            drugE, seE, x = self.model.forward(bioLoader5P.x,
                                               bioLoader5P.drugGraphData.edge_index, bioLoader5P.seGraphData.edge_index,
                                               bioLoader5P.drugTrainNodeIds,
                                               bioLoader5P.seNodeIds,
                                               bioLoader5P.proteinNodeIds,
                                               bioLoader5P.drugId2ProteinIndices,
                                               bioLoader5P.graphBatch,
                                               bioLoader5P.nDrug,
                                               bioLoader5P.drugFeatures
                                               )

            err, out = self.getMatDotLoss(drugE, seE, bioLoader5P.trainOutMatrix)
            err.backward()
            optimizer.step()

            # Eval:
            if debug and epoch % 10 == 0:
                out2 = out.detach().numpy()

                # torch.no_grad()

                self.logger.infoAll((out2.shape, target2.shape, np.sum(out2), np.sum(target2)))

                auc2 = roc_auc_score(target2.reshape(-1), out2.reshape(-1))
                aupr2 = average_precision_score(target2.reshape(-1), out2.reshape(-1))
                self.logger.infoAll(("Error: ", err))
                self.logger.infoAll(("Train: AUC, AUPR: ", auc2, aupr2))

                outValid = self.model.cal(x[bioLoader5P.drugValidateNodeIds], x[bioLoader5P.seNodeIds])
                outValid = outValid.detach().numpy()

                aucv = roc_auc_score(validateOutput.reshape(-1), outValid.reshape(-1))
                auprv = average_precision_score(validateOutput.reshape(-1), outValid.reshape(-1))
                errValid = getF2Erros(validateOutput, outValid)
                allValErros.append(errValid)

                self.logger.infoAll(("Val: AUC, AUPR, Erros: ", aucv, auprv, errValid))

                outTest = self.model.cal(x[bioLoader5P.drugTestNodeIds], x[bioLoader5P.seNodeIds])
                outTest = outTest.detach().numpy()

                auc = roc_auc_score(targetTest.reshape(-1), outTest.reshape(-1))
                aupr = average_precision_score(targetTest.reshape(-1), outTest.reshape(-1))
                errTest = getF2Erros(targetTest, outTest)
                allTestErros.append(errTest)

                self.logger.infoAll(("Test: AUC, AUPR, Erros: ", auc, aupr, errTest))

                allEval.append([aucv, auprv, errValid, auc, aupr, errTest])
                allOuts.append([out2, outTest])

                # torch.enable_grad()

        if pred and not debug:
            outTest = self.model.cal(x[bioLoader5P.drugTestNodeIds], x[bioLoader5P.seNodeIds])
            outTest = outTest.detach().numpy()
        if pred:
            # aucvs = []
            # for i in range(len(allEval)):
            #     aucvs.append(allEval[i][0])
            # aucvs = np.asarray(aucvs)
            # idMax = np.argmax(aucvs)

            idMax = np.argmin(allValErros)

            out2, outTest = allOuts[idMax]
            print("\nValidation: ", allEval[idMax])

            return out2, outTest
        else:
            return drugE, seE

    def fitAndPredict(self, bioLoader):
        self.resetModel(numAtomFeature=bioLoader.N_ATOMFEATURE)
        out2, outTest = self.train(bioLoader, debug=True)
        self.repred = out2
        return outTest

    def getParams(self):
        return "MPNNX"


def getF2Erros(pred, target):
    v = pred - target
    v = v * v
    return np.sum(v)


if __name__ == "__main__":
    pass
