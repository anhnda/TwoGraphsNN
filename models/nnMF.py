import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import config
import numpy as np
import math
from sklearn.decomposition import NMF

from torch.nn.modules.loss import MSELoss
from sklearn.metrics import roc_auc_score, average_precision_score


class NNMF(nn.Module):
    def __init__(self):
        super(NNMF, self).__init__()

    def feedData(self, input):
        # self.U = None
        # self.V = None
        # self.Y = None
        # self.XC = None
        nD, nV = input.shape
        self.inp = torch.from_numpy(input).float()

        self.U = Parameter(torch.zeros((nD, config.EMBED_DIM), dtype=torch.float), requires_grad=True)
        self.__my_reset_uniform(self.U, 0.0001, 0.3)

        self.V = Parameter(torch.zeros((config.EMBED_DIM, nV), dtype=torch.float), requires_grad=True)
        self.__my_reset_uniform(self.V, 0.0001, 0.3)

    def getParams(self):
        return {"LAMBDA_2": config.LAMBDA_2, "R1": config.LAMBDA_R1, "R2": config.LAMBDA_R2, "R12": config.LAMBDA_R12}

    def getLoss(self, stype=config.DICT_SPARSE):
        loss = 0

        R = torch.matmul(self.U, self.V)
        err1 = R - self.inp
        err1 = self.__getF2Err(err1)
        loss += err1

        reg1 = self.__getF2Err(self.U) + self.__getF2Err(self.V)
        reg1 *= 0.01
        loss += reg1

        return loss

    def predict(self):
        x = torch.matmul(self.U, self.V)
        return x.detach().numpy()

    def fit(self, data):
        self.feedData(data)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

        for i in range(config.N_EPOCH_2):
            optimizer.zero_grad()
            loss = self.getLoss()
            loss.backward()
            optimizer.step()
            print("\r%s" % i, end="")
            if i % 10 == 0:
                re = self.predict()

                auc = roc_auc_score(data.reshape(-1), re.reshape(-1))
                aupr = average_precision_score(data.reshape(-1), re.reshape(-1))
                print("\nLoss: ", loss)
                print("\nAUC, AUPR:", auc, aupr)

    def __getF2Err(self, err):
        err = torch.mul(err, err)
        err = torch.sum(err)
        # err =  self.mseLoss(err,self.zeroTargets)
        return err

    def __my_reset_parameters(self, tensorobj):
        stdv = 1. / math.sqrt(tensorobj.size(1))
        tensorobj.data.uniform_(-stdv, stdv)

    def __my_reset_uniform(self, tensorobj, min=0, max=1):
        tensorobj.data.uniform_(min, max)


if __name__ == "__main__":
    from dataFactory.loader import BioLoader

    bioLoader = BioLoader()
    trainPath = bioLoader.getPathIFold(1)
    bioLoader.createTrainTestGraph(trainPath)
    data = bioLoader.trainOutMatrix.numpy()

    model = NNMF()
    model.fit(data)
