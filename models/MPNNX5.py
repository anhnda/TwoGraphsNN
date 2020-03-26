from models.net5 import Net5
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import numpy as np

from utils import utils
from utils.logger.logger2 import MyLogger
import inspect


class MPNNX5:
    def __init__(self):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logPath = "%s/logs/MPNNX5_%s" % (config.C_DIR, utils.getCurrentTimeString())
        print("Logging path: ", logPath)
        self.logger = MyLogger(logPath)

        self.logger.infoAll(inspect.getsource(Net5))

        self.logger.infoAll(("Undirected graph: ", config.UN_DIRECTED))
        self.name = "MPNNX5"
        self.isFitAndPredict = True

    def resetModel(self, numAtomFeature):
        self.net5 = Net5(numNode=config.MAX_NODE + 1, numAtomFeature=numAtomFeature)
        self.model = self.net5.to(self.device)

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

    def train(self, bioLoader5, debug=True, pred=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss = torch.nn.MSELoss()
        # loss = torch.nn.BCELoss()

        for epoch in range(config.N_EPOCH):

            print("\r%s" % epoch, end="")
            optimizer.zero_grad()
            drugE, seE, x = self.model.forward(bioLoader5.x,
                                               bioLoader5.drugGraphData.edge_index, bioLoader5.seGraphData.edge_index,
                                               bioLoader5.drugTrainNodeIds,
                                               bioLoader5.seNodeIds,
                                               bioLoader5.graphBatch,
                                               bioLoader5.nDrug
                                               )

            err, out = self.getMatDotLoss(drugE, seE, bioLoader5.trainOutMatrix)

            out2 = out.detach().numpy()
            target2 = bioLoader5.trainOutMatrix.numpy()

            err.backward()
            optimizer.step()

            # Eval:
            if debug and epoch % 10 == 0:
                # torch.no_grad()

                self.logger.infoAll((out2.shape, target2.shape, np.sum(out2), np.sum(target2)))

                auc2 = roc_auc_score(target2.reshape(-1), out2.reshape(-1))
                aupr2 = average_precision_score(target2.reshape(-1), out2.reshape(-1))
                self.logger.infoAll(("Error: ", err))
                self.logger.infoAll(("Train: AUC, AUPR: ", auc2, aupr2))

                # outTest = torch.matmul(x[bioLoader2.drugTestNodeIds], x[bioLoader2.seNodeIds].t())
                outTest = self.model.cal(x[bioLoader5.drugTestNodeIds], x[bioLoader5.seNodeIds])
                outTest = outTest.detach().numpy()

                targetTest = bioLoader5.testOutMatrix.numpy()

                auc = roc_auc_score(targetTest.reshape(-1), outTest.reshape(-1))
                aupr = average_precision_score(targetTest.reshape(-1), outTest.reshape(-1))
                self.logger.infoAll(("Test: AUC, AUPR: ", auc, aupr))
                # torch.enable_grad()

                if epoch > 10:
                    np.savetxt("out/drugE.txt", drugE.detach().numpy())
                    np.savetxt("out/seE.txt", seE.detach().numpy())

        if pred and not debug:
            # outTest = torch.matmul(x[bioLoader2.drugTestNodeIds], x[bioLoader2.seNodeIds].t())
            outTest = self.model.cal(x[bioLoader5.drugTestNodeIds], x[bioLoader5.seNodeIds])
            outTest = outTest.detach().numpy()
        if pred:
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


if __name__ == "__main__":
    pass