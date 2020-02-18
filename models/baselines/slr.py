import torch
import numpy as np
from torch.nn.parameter import Parameter
import config
#
#
# class SLR(torch.nn.Module):
#     def __init__(self):
#         super(SLR, self).__init__()
#
#     def fit(self, x, y):
#         nSample, nFeature = x.shape
#
#         w = torch.zeros([nFeature])
#         self.w = Parameter(w, requires_grad=True)
#         c = torch.ones(1)
#         self.c = Parameter(c, requires_grad=True)
#
#         self.__my_reset_uniform(self.w, 0.001, 0.03)
#         self.x = torch.from_numpy(x).float()
#         self.y = torch.from_numpy(y).float()
#         self.__learn()
#
#     def predict(self, x):
#         # if type(x) == np.ndarray:
#         # torch.no_grad()
#         x = torch.from_numpy(x).float()
#         v = torch.matmul(x, self.w) + self.c
#         v = -v
#         v2 = torch.exp(v) + 1
#         v2 = 1.0 / v2
#         # torch.enable_grad()
#
#         return v2.detach().numpy()
#
#     def fitEval(self, x, y, xtest, ytest):
#         from evaluator import getAUCAUPR
#         nSample, nFeature = x.shape
#
#         w = torch.zeros([nFeature])
#         self.w = Parameter(w, requires_grad=True)
#         c = torch.ones(1)
#         self.c = Parameter(c, requires_grad=True)
#
#         self.__my_reset_uniform(self.w, 0.001, 0.03)
#         self.x = torch.from_numpy(x).float()
#         self.y = torch.from_numpy(y).float()
#
#         torch.autograd.set_detect_anomaly(True)
#         optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
#         # nTrain = inputTrain.shape[0]
#         # nBatch = nTrain / config.BATCH_SIZE
#         for i in range(config.N_EPOCH_SLR):
#             # For Python 2
#             print("\r%s" % i, end="")
#             optimizer.zero_grad()
#             loss = self.__getLoss()
#             loss.backward()
#             optimizer.step()
#             if i % 50 == 0:
#                 pred = self.predict(xtest)
#                 print("\t", getAUCAUPR(ytest, pred))
#
#     def __getLoss(self):
#         v = -(torch.matmul(self.x, self.w) + self.c)
#         v = torch.mul(v, self.y)
#         v = torch.exp(v)
#         v = v + 1
#         v = torch.log(v)
#         e1 = torch.sum(v)
#         e2 = torch.sum(torch.abs(self.w))
#         v = e1 * config.LR_C + e2
#         return v
#
#     def __project(self):
#         self.w.data[self.w > 0] = 0
#
#     def __learn(self):
#         torch.autograd.set_detect_anomaly(True)
#         optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
#         # nTrain = inputTrain.shape[0]
#         # nBatch = nTrain / config.BATCH_SIZE
#         for i in range(config.N_EPOCH_SLR):
#             # print("\r%s" % i, end="")
#             optimizer.zero_grad()
#             loss = self.__getLoss()
#             loss.backward()
#             optimizer.step()
#             if config.NONE_NEGRATIVE_SLR:
#                 # self.__project()
#                 pass
#
#     def __my_reset_uniform(self, tensorobj, min=0, max=1):
#         tensorobj.data.uniform_(min, max)
#
#
# class MultiSLR(torch.nn.Module):
#     def __init__(self):
#         super(MultiSLR, self).__init__()
#
#     def fit(self, x, y):
#         nSample, nFeature = x.shape
#         _, nClass = y.shape
#
#         w = torch.zeros([nFeature, nClass])
#         self.w = Parameter(w, requires_grad=True)
#         c = torch.ones([nClass])
#         self.c = Parameter(c, requires_grad=True)
#
#         self.__my_reset_uniform(self.w, 0.001, 0.03)
#         self.x = torch.from_numpy(x).float()
#         self.y = torch.from_numpy(y).float()
#         self.__learn()
#
#     def predict(self, x):
#         # if type(x) == np.ndarray:
#         # torch.no_grad()
#         x = torch.from_numpy(x).float()
#         v = torch.matmul(x, self.w) + self.c
#         v = -v
#         v2 = torch.exp(v) + 1
#         v2 = 1.0 / v2
#         # torch.enable_grad()
#
#         return v2.detach().numpy()
#
#     def fitEval(self, x, y, xtest, ytest):
#         from evaluator import getAUCAUPR
#         nSample, nFeature = x.shape
#
#         w = torch.zeros([nFeature])
#         self.w = Parameter(w, requires_grad=True)
#         c = torch.ones(1)
#         self.c = Parameter(c, requires_grad=True)
#
#         self.__my_reset_uniform(self.w, 0.001, 0.03)
#         self.x = torch.from_numpy(x).float()
#         self.y = torch.from_numpy(y).float()
#
#         torch.autograd.set_detect_anomaly(True)
#         optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
#         # nTrain = inputTrain.shape[0]
#         # nBatch = nTrain / config.BATCH_SIZE
#         for i in range(config.N_EPOCH_SLR):
#             # For Python 2
#             print("\r%s" % i, end="")
#             optimizer.zero_grad()
#             loss = self.__getLoss()
#             loss.backward()
#             optimizer.step()
#             if i % 50 == 0:
#                 pred = self.predict(xtest)
#                 print("\t", getAUCAUPR(ytest, pred))
#
#     def __getLoss(self):
#         v = -(torch.matmul(self.x, self.w) + self.c)
#         v = torch.mul(v, self.y)
#         v = torch.exp(v)
#         v = v + 1
#         v = torch.log(v)
#         e1 = torch.sum(v, dim=0)
#         e2 = torch.sum(torch.abs(self.w), dim=0)
#         # e2 = torch.sum(torch.mul(self.w, self.w), dim=0)
#         v = e1 * config.LR_C + e2
#         v = torch.sum(v)
#         return v
#
#     def __project(self):
#         self.w.data[self.w < 0] = 0
#
#     def __learn(self):
#         torch.autograd.set_detect_anomaly(True)
#         optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
#         # nTrain = inputTrain.shape[0]
#         # nBatch = nTrain / config.BATCH_SIZE
#         for i in range(config.N_EPOCH_SLR):
#             print("\r%s" % i, end="")
#             optimizer.zero_grad()
#             loss = self.__getLoss()
#             loss.backward()
#             optimizer.step()
#             if config.NONE_NEGRATIVE_SLR:
#                 self.__project()
#                 pass
#
#     def __my_reset_uniform(self, tensorobj, min=0, max=1):
#         tensorobj.data.uniform_(min, max)


class PSLR(torch.nn.Module):
    def __init__(self):
        super(PSLR, self).__init__()

    def fit(self, x, y, ppw):
        nSample, nFeature = x.shape
        _, nClass = y.shape

        nR, nA = ppw.shape

        w = torch.zeros([nFeature, nClass])
        self.w = Parameter(w, requires_grad=True)

        self.wc = self.w[:config.CHEM_FINGERPRINT_SIZE, :]
        self.wr = self.w[config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + nR, :]
        self.wa = self.w[config.CHEM_FINGERPRINT_SIZE + nR:, :]

        c = torch.ones([nClass])
        self.c = Parameter(c, requires_grad=True)

        self.__my_reset_uniform(self.w, 0.001, 0.03)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.__learn()

    def predict(self, x):
        # if type(x) == np.ndarray:
        # torch.no_grad()
        x = torch.from_numpy(x).float()
        v = torch.matmul(x, self.w) + self.c
        v = -v
        v2 = torch.exp(v) + 1
        v2 = 1.0 / v2
        # torch.enable_grad()

        return v2.detach().numpy()

    def fitEval(self, x, y, xtest, ytest):
        from evaluator import getAUCAUPR
        nSample, nFeature = x.shape

        w = torch.zeros([nFeature])
        self.w = Parameter(w, requires_grad=True)
        c = torch.ones(1)
        self.c = Parameter(c, requires_grad=True)

        self.__my_reset_uniform(self.w, 0.001, 0.03)
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
        # nTrain = inputTrain.shape[0]
        # nBatch = nTrain / config.BATCH_SIZE
        for i in range(config.N_EPOCH_SLR):
            # For Python 2
            print("\r%s" % i, end="")
            optimizer.zero_grad()
            loss = self.__getLoss()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                pred = self.predict(xtest)
                print("\t", getAUCAUPR(ytest, pred))
    def getW(self):
        self.W = self.w.detach().numpy()
        return self.W
    def __getLoss(self):

        # v = -(torch.matmul(self.x, self.w) + self.c)
        # v = torch.mul(v, self.y)
        # v = torch.exp(v)
        # v = v + 1
        # v = torch.log(v)

        z = -(torch.matmul(self.x, self.w) + self.c)
        z = torch.exp(z) + 1 + 1e-5
        z = 1 / z

        # print(z.shape, (z == 1).nonzero())

        z1 = torch.mul(torch.log(z), self.y)
        z2 = torch.mul(torch.log(1 - z), 1 - self.y)

        v = -z1 - z2
        e1 = torch.sum(v, dim=0)
        # e2 = torch.sum(torch.abs(self.w), dim=0)

        # e2 = torch.sum(torch.mul(self.w, self.w), dim=0)

        # e2 = torch.sum(torch.abs(self.w), dim=0) - config.LAMBDA_R12 * (
        #           self.__getMaxErr2(self.wc) + self.__getMaxErr2(self.wr) + self.__getMaxErr2(self.wa))
        e2 = torch.sum(self.getExGLasso(self.wc) + self.getExGLasso(self.wr) + self.getExGLasso(self.wa))
        v = e1 * config.LR_C + e2
        v = torch.sum(v)
        return v

    def __getMaxErr2(self, v):
        return torch.abs(torch.max(v, dim=0)[0]) * v.shape[0]
    def getExGLasso(self, v):
        v = torch.sum(torch.abs(v), dim=1)[0]
        return v * v

    def __project(self):
        self.w.data[self.w < 0] = 0

    def __learn(self):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.parameters(), lr=config.LEARNING_RATE_SLR)
        # nTrain = inputTrain.shape[0]
        # nBatch = nTrain / config.BATCH_SIZE
        for i in range(config.N_EPOCH_SLR):
            print("\r%s" % i, end="")
            optimizer.zero_grad()
            loss = self.__getLoss()
            loss.backward()
            optimizer.step()
            if config.NONE_NEGRATIVE_SLR:
                self.__project()
                pass

    def __my_reset_uniform(self, tensorobj, min=0, max=1):
        tensorobj.data.uniform_(min, max)
