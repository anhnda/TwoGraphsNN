from models.G3N2Level import G3N2Level
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import numpy as np

from utils import utils
from utils.logger.logger2 import MyLogger
import inspect
from sklearn.metrics import mean_squared_error as mse


class WrapperG3N2Level:
    def __init__(self, logger=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.name = "G3N2Level"
        self.isFitAndPredict = True

    def setLogger(self, logger):
        self.logger = logger
        self.logger.infoAll(inspect.getsource(G3N2Level))
        self.logger.infoAll(("Num Graph Layer: ", config.N_LAYER_LEVEL_1, config.N_LAYER_LEVEL_2))
        self.logger.infoAll(("Layer TYPE:", config.LEVEL_1_LAYER, config.LEVEL_2_LAYER))

    def resetModel(self, outSize):
        self.net = G3N2Level(outSize, layerType1=config.LEVEL_1_LAYER, layerType2 = config.LEVEL_2_LAYER, numNode=config.MAX_NODE + 1)
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

    def train(self, bioLoader5P):
        if config.OPTIMIZER == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        else:
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=0.01)

        from dataFactory.nestedGraph import convertBioLoaderToNestedGraph

        nestedGraphData = convertBioLoaderToNestedGraph(bioLoader5P)
        graphBatch1, numGraph = nestedGraphData.createBatchForLevel(1)
        graphBatch1o = bioLoader5P.graphBatch
        assert torch.all(torch.eq(graphBatch1.edge_index, graphBatch1o.edge_index))
        assert torch.all(torch.eq(graphBatch1.x, graphBatch1o.x))
        assert torch.all(torch.eq(graphBatch1.batch, graphBatch1o.batch))

        graph2 = nestedGraphData.getGraphListAtLevel(2)[0]


        trainIds = torch.from_numpy(np.asarray(bioLoader5P.drugTrainIdList)).long()
        testIds = torch.from_numpy(np.asarray(bioLoader5P.drugTestIdList)).long()
        valIds = torch.from_numpy(np.asarray(bioLoader5P.drugValidateIdList)).long()

        lossFunc = torch.nn.MSELoss()

        allOuts = []
        allEval = []
        allValErros = []
        allTestErros = []

        trainTargetTensor = bioLoader5P.trainOutMatrix
        trainTargetNumpy = trainTargetTensor.numpy()
        validateTargetNumpy = bioLoader5P.validOutMatrix.numpy()
        testTargetNumpy = bioLoader5P.testOutMatrix.numpy()

        for epoch in range(config.N_EPOCH):

            print("\r%s" % epoch, end="")
            optimizer.zero_grad()
            x = self.model.forward(graphBatch1, graph2)
            out = self.model.calOut(x, trainIds)
            err = lossFunc(out, trainTargetTensor)
            err.backward()
            optimizer.step()

            # Eval:
            if epoch % 10 == 0:
                out2 = out.detach().numpy()

                # torch.no_grad()

                self.logger.infoAll((out2.shape, trainTargetNumpy.shape, np.sum(out2), np.sum(trainTargetNumpy)))

                auc2 = roc_auc_score(trainTargetNumpy.reshape(-1), out2.reshape(-1))
                aupr2 = average_precision_score(trainTargetNumpy.reshape(-1), out2.reshape(-1))
                self.logger.infoAll(("Error: ", err))
                self.logger.infoAll(("Train: AUC, AUPR: ", auc2, aupr2))

                outValid = self.model.calOut(x, valIds)
                outValid = outValid.detach().numpy()

                aucv = roc_auc_score(validateTargetNumpy.reshape(-1), outValid.reshape(-1))
                auprv = average_precision_score(validateTargetNumpy.reshape(-1), outValid.reshape(-1))
                errValid = mse(validateTargetNumpy, outValid)
                allValErros.append(errValid)

                self.logger.infoAll(("Val: AUC, AUPR, Erros: ", aucv, auprv, errValid))

                outTest = self.model.calOut(x, testIds)
                outTest = outTest.detach().numpy()

                auc = roc_auc_score(testTargetNumpy.reshape(-1), outTest.reshape(-1))
                aupr = average_precision_score(testTargetNumpy.reshape(-1), outTest.reshape(-1))
                errTest = mse(testTargetNumpy, outTest)
                allTestErros.append(errTest)

                self.logger.infoAll(("Test: AUC, AUPR, Erros: ", auc, aupr, errTest))

                allEval.append([aucv, auprv, errValid, auc, aupr, errTest])
                allOuts.append([out2, outTest])

                # torch.enable_grad()

        idMax = np.argmin(allValErros)
        out2, outTest = allOuts[idMax]
        print("\nValidation: ", allEval[idMax])

        return out2, outTest

    def fitAndPredict(self, bioLoader):
        self.resetModel(bioLoader.nSe)
        out2, outTest = self.train(bioLoader)
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
