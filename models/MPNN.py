from dataFactory.loader import BioLoader
from models.net import Net
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config

import numpy as np

from utils import utils
from utils.logger.logger2 import MyLogger
import inspect
class MPNNWrapper:
    def __init__(self, iFold=1):
        self.bioLoader = BioLoader()
        self.trainPath = BioLoader.getPathIFold(iFold)
        self.bioLoader.createTrainTestGraph(self.trainPath)
        self.net = Net(numNode=self.bioLoader.NUM_NODES + 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.net.to(self.device)

        logPath = "%s/logs/MPNN_%s" % (config.C_DIR, utils.getCurrentTimeString())
        print("Logging path: ", logPath)
        self.logger = MyLogger(logPath)
        self.logger.infoAll(self.model)
        self.logger.infoAll(inspect.getsource(Net.forward))

        self.logger.infoAll(self.trainPath)


    def __getF2Err(self, err):
        err = torch.mul(err, err)
        err = torch.sum(err)
        # err =  self.mseLoss(err,self.zeroTargets)
        return err

    def getMatDotLoss(self, u, v, t):

        loss = 0
        out = torch.matmul(u, v.t())
        # out = torch.sigmoid(out)
        err1 = out - t
        err1 = self.__getF2Err(err1)
        loss += err1

        reg1 = self.__getF2Err(u) + self.__getF2Err(v)
        reg1 *= 0.01
        loss += reg1

        return loss, out

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss = torch.nn.MSELoss()
        # loss = torch.nn.BCELoss()

        for epoch in range(config.N_EPOCH):

            print("\r%s" % epoch, end="")
            optimizer.zero_grad()
            drugE, seE, x = self.model.forward(self.bioLoader.graphData, self.bioLoader.drugTrainNodeIds,
                                               self.bioLoader.seNodeIds)
            err, out = self.getMatDotLoss(drugE, seE, self.bioLoader.trainOutMatrix)

            out2 = out.detach().numpy()
            target2 = self.bioLoader.trainOutMatrix.numpy()

            err.backward()
            optimizer.step()

            # Eval:
            if epoch % 10 == 0:
                # torch.no_grad()

                self.logger.infoAll((out2.shape, target2.shape, np.sum(out2), np.sum(target2)))

                auc2 = roc_auc_score(target2.reshape(-1), out2.reshape(-1))
                aupr2 = average_precision_score(target2.reshape(-1), out2.reshape(-1))
                self.logger.infoAll(("Error: ", err))
                self.logger.infoAll(("Train: AUC, AUPR: ", auc2, aupr2))


                outTest = torch.matmul(x[self.bioLoader.drugTestNodeIds], x[self.bioLoader.seNodeIds].t())
                outTest = outTest.detach().numpy()
                targetTest = self.bioLoader.testOutMatrix.numpy()

                auc = roc_auc_score(targetTest.reshape(-1), outTest.reshape(-1))
                aupr = average_precision_score(targetTest.reshape(-1), outTest.reshape(-1))
                self.logger.infoAll(("Test: AUC, AUPR: ", auc, aupr))
                # torch.enable_grad()

            if epoch == config.N_EPOCH - 1:
                np.savetxt("out/trainPred.txt", out2)
                np.savetxt("out/trainTarget.txt", target2)


if __name__ == "__main__":
    mpnn = MPNNWrapper()
    mpnn.train()
