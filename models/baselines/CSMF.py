import config
import numpy as np
from utils import utils
from sklearn.metrics import roc_auc_score, average_precision_score


class CSMF:
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "CSMF"
        self.repred = None
        self.model = "CSMF"
        pass

    def feed(self, input, output):
        self.X = input
        self.Y = output
        self.nTrain, self.nDim = self.X.shape
        self.nSE = output.shape[1]

        # DrugSimMatrix
        self.M = np.zeros((self.nTrain, self.nTrain))
        for i in range(self.nTrain):
            for j in range(i, self.nTrain):
                v = utils.getTanimotoScore(self.X[i], self.X[j])
                self.M[i, j] = self.M[j, i] = v

        self.F = np.random.uniform(0.00, 1.0 / (config.EMBED_DIM * config.EMBED_DIM),
                                   (self.nTrain, config.EMBED_DIM))
        self.G = np.random.uniform(0.00, 1.0 / (config.EMBED_DIM * config.EMBED_DIM), (self.nSE, config.EMBED_DIM))

        # self.F.fill(1.0 / (config.NUM_FACTORS*config.NUM_FACTORS))
        # self.G.fill(1.0 / (config.NUM_FACTORS*config.NUM_FACTORS))

        # self.M = np.loadtxt("%s%s/%s"%(config.FOLD_PREFIX,iFold,config.DRUG_SIM_FILE))
        # self.N = self.loadSESim()
        # self.M = np.ones((self.NUM_DRUG,self.NUM_DRUG),dtype=float)

        self.N = np.ones((self.nSE, self.nSE), dtype=float)
        self.sumAdaGraF = np.zeros((self.nTrain, config.EMBED_DIM))
        self.sumAdaGraG = np.zeros((self.nSE, config.EMBED_DIM))

        self.normM(self.M)
        self.normM(self.N)

        print("Finished initialization")

    def fit(self, input, output):
        self.feed(input, output)
        self.learn()

    def predict(self, inputTest):

        def calSimTestTrainMatrix():
            from utils import utils
            inputTrain = self.X
            nTrain = inputTrain.shape[0]
            nTest = inputTest.shape[0]
            simMatrix = np.ones((nTest, nTrain))
            for ii in range(nTest):
                for j in range(nTrain):
                    sc = utils.getTanimotoScore(inputTest[ii], inputTrain[j])
                    simMatrix[ii][j] = sc
            args = np.argsort(simMatrix, axis=1)
            args = args[:, -config.CS_KNN:]
            weights = []
            for ii in range(nTest):
                weight = simMatrix[ii, args[ii]]
                weight /= sum(weight)
                weights.append(weight)

            weights = np.vstack(weights)

            SimIndices = args
            SimWeights = weights
            return SimIndices, SimWeights

        SimIndices, SimWeights = calSimTestTrainMatrix()
        initKNN = []
        nTest = SimIndices.shape[0]
        for i in range(nTest):
            ar = self.F[SimIndices[i], :]
            ar = np.dot(SimWeights[i], ar)
            initKNN.append(ar)
        initKNN = np.vstack(initKNN)
        v = np.dot(initKNN, self.G.transpose())
        v = self.normEXP(v)
        return v

    def fitAndPredict(self, input, output, inputTest):
        self.fit(input, output)
        self.repred = self.getP()
        return self.predict(inputTest)

    def doIteration(self):
        P = self.getP()
        P -= self.Y
        # calculate dF
        dF = np.dot(P, self.G)

        v = 2 * config.LAMBDA_R * self.F
        dF += v
        v = 2 * config.LAMBDA_M * self.M
        v = np.dot(v, self.F)
        dF += v

        # calculate dG
        dG = np.dot(P.transpose(), self.F)

        v = 2 * config.LAMBDA_R * self.G
        dG += v
        v = 2 * config.LAMBDA_N * self.N
        v = np.dot(v, self.G)
        dG += v

        self.sumAdaGraF += np.multiply(dF, dF)
        self.sumAdaGraG += np.multiply(dG, dG)

        normGraF = np.sqrt(self.sumAdaGraF)
        normGraF = 1.0 / normGraF

        normGraG = np.sqrt(self.sumAdaGraG)
        normGraG = 1.0 / normGraG

        # Gradient descent
        dF *= config.LEARNING_RATE_CS
        dG *= config.LEARNING_RATE_CS

        dF = np.multiply(dF, normGraF)
        dG = np.multiply(dG, normGraG)

        self.F -= dF
        self.G -= dG

    def myNMF(self):
        def doIter():
            r = self.getDotR()
            r -= self.Y
            dG = np.dot(self.F.transpose(), r).transpose()
            dF = np.dot(r, self.G)
            dG *= config.LEARNING_RATE_CS
            dF *= config.LEARNING_RATE_CS
            self.F -= dF
            self.G -= dG

        for i in range(config.NUM_ITER_CS):
            doIter()

    def learn(self):

        for i in range(config.NUM_ITER_CS):
            print("\r%s" % i, end="")
            # if i % 15 == 0 and i > 0:
            #     # self.doEval()
            #     auc, aupr = self.doEvalDrugSlice()
            #     if i == 405:
            #         self.logger.infoFile((auc, aupr))

            self.doIteration()
        print("\nDone")

    def getDotR(self):
        return np.dot(self.F, self.G.transpose())

    def normY(self):
        expY = np.exp(self.Y)
        v = expY + 1
        expY /= v
        self.Y = expY

    def normEXP(self, m):
        expY = np.exp(m)
        v = expY + 1
        expY /= v
        return m

    def getP(self):
        v = np.dot(self.F, self.G.transpose())
        v = np.exp(v)
        v2 = v + 1
        v /= v2
        return v

    def normM(self, m):
        sum_row = np.sum(m, axis=1)
        diag = np.diag(sum_row)
        m -= diag
        m *= -1
        return m
    def getParams(self):
        return "CSMF_P"


if __name__ == "__main__":
    from dataFactory.loader2 import BioLoader2
    iFold = 1
    bioLoader2 = BioLoader2()
    trainPath = BioLoader2.getPathIFold(iFold)
    bioLoader2.createTrainTestGraph(trainPath)

    from models.baselines.omodels import LNSM


    model = CSMF()
    # model = LNSM()
    outTest = model.fitAndPredict(bioLoader2.trainInpMat, bioLoader2.trainOutMatrix.numpy(), bioLoader2.testInpMat)
    targetTest = bioLoader2.testOutMatrix.numpy()
    # print (targetTest, outTest)
    auc = roc_auc_score(targetTest.reshape(-1), outTest.reshape(-1))
    aupr = average_precision_score(targetTest.reshape(-1), outTest.reshape(-1))
    print ("AUC, AUPR: ", auc, aupr)

