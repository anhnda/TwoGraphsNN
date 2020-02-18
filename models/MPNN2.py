from dataFactory.loader2 import BioLoader2
from models.net2 import Net2
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import numpy as np

from utils import utils
from utils.logger.logger2 import MyLogger
import inspect
class MPNNWrapper2:
    def __init__(self, iFold=1):
        self.bioLoader2 = BioLoader2()
        self.trainPath = BioLoader2.getPathIFold(iFold)
        self.bioLoader2.createTrainTestGraph(self.trainPath)
        self.net2 = Net2(numNode=self.bioLoader2.NUM_NODES + 1)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.net2.to(self.device)
        logPath = "%s/logs/MPNN2_%s" % (config.C_DIR, utils.getCurrentTimeString())
        print("Logging path: ", logPath)
        self.logger = MyLogger(logPath)
        self.logger.infoAll(self.model)
        self.logger.infoAll(inspect.getsource(Net2.forward))

        self.logger.infoAll(self.trainPath)
        self.logger.infoAll(("Undirected graph: ", config.UN_DIRECTED))
        self.name = "MPNN"



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
        out = torch.matmul(u, v.t())
        # out = torch.sigmoid(out)
        err1 = out - t
        err1 = self.__getF2Err(err1)
        loss += err1

        reg1 = self.__getF1Err(u) + self.__getF1Err(v)
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
            drugE, seE, x = self.model.forward(self.bioLoader2.x,
                                               self.bioLoader2.drugGraphData.edge_index, self.bioLoader2.seGraphData.edge_index,
                                               self.bioLoader2.drugTrainNodeIds,
                                               self.bioLoader2.seNodeIds)
            err, out = self.getMatDotLoss(drugE, seE, self.bioLoader2.trainOutMatrix)

            out2 = out.detach().numpy()
            target2 = self.bioLoader2.trainOutMatrix.numpy()

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

                outTest = torch.matmul(x[self.bioLoader2.drugTestNodeIds], x[self.bioLoader2.seNodeIds].t())
                outTest = outTest.detach().numpy()
                targetTest = self.bioLoader2.testOutMatrix.numpy()

                auc = roc_auc_score(targetTest.reshape(-1), outTest.reshape(-1))
                aupr = average_precision_score(targetTest.reshape(-1), outTest.reshape(-1))
                self.logger.infoAll(("Test: AUC, AUPR: ", auc, aupr))
                # torch.enable_grad()

                if epoch > 10:
                    np.savetxt("out/drugE.txt", drugE.detach().numpy())
                    np.savetxt("out/seE.txt", seE.detach().numpy())


if __name__ == "__main__":
    mpnn = MPNNWrapper2()
    mpnn.train()
