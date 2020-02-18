import numpy as np
from sklearn.linear_model import LogisticRegression
from models.baselines  import lnsm
import config
from utils import utils
from sklearn import svm
from sklearn.neural_network import MLPRegressor, MLPClassifier

import sys, time

def checkOneClass(inp, nSize):
    s = sum(inp)
    if s == 0:
        ar = np.zeros(nSize)
    elif s == len(inp):
        ar = np.ones(nSize)
    else:
        ar = -1
    return ar


class LNSM:
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "LNSM"
        self.repred = None
        self.model = "LNSM"
    def getParams(self):
        return "Default"
    def fit(self, input, output):
        self.inputTrain = input
        self.Y = lnsm.learnLNSM(input, output)
        # self.repred = output

    def predict(self, inputTest):
        preds = []
        for v in inputTest:
            w = lnsm.getRowLNSM(v, self.inputTrain, -1)
            pred = np.matmul(w, self.Y)
            preds.append(pred)
        preds = np.vstack(preds)
        return preds

    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        self.fit(intpuTrain, outputTrain)
        self.repred = self.predict(intpuTrain)
        return self.predict(inputTest)


#
# class MultiSLRModel:
#
#     def __init__(self):
#         self.isFitAndPredict = True
#         self.name = "MultiSLR"
#         self.repred = None
#
#     def fitAndPredict(self, inputTrain, outputTrain, inputTest):
#         import warnings
#         from sklearn.exceptions import ConvergenceWarning
#         from models.baselines.slr import MultiSLR
#         warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
#
#         print(inputTrain.shape, outputTrain.shape, inputTest.shape)
#
#         nClass = outputTrain.shape[1]
#         outputs = []
#         reps = []
#         nTest = inputTest.shape[0]
#         print("MSLR for %s classes" % nClass)
#         self.model = MultiSLR()
#         self.model.fit(inputTrain, outputTrain)
#         outputs = self.model.predict(inputTest)
#         # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
#         self.repred = self.model.predict(inputTrain)
#
#         print(outputs.shape)
#         print("\nDone")
#         return outputs
#
#     def getInfo(self):
#         return self.model


class PSLRModel:

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "PSLR"
        self.repred = None
    def getParams(self):
        return {"LR_C": config.LR_C, "R12": config.LAMBDA_R12}
    def fitAndPredict(self, inputTrain, outputTrain, ppw, inputTest ):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from models.baselines.slr import PSLR
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("PSLR for %s classes" % nClass)
        self.model = PSLR()
        self.model.fit(inputTrain, outputTrain, ppw)
        outputs = self.model.predict(inputTest)
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = self.model.predict(inputTrain)

        print(outputs.shape)
        print("\nDone")
        return outputs

    def fit(self, inputTrain, outputTrain, ppw):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from models.baselines.slr import PSLR
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

        print(inputTrain.shape, outputTrain.shape)

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        print("PSLR for %s classes" % nClass)
        self.model = PSLR()
        self.model.fit(inputTrain, outputTrain, ppw)
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        print("\nDone")

    def savePattern(self, suffix="X"):
        w = self.model.getW()
        np.savetxt("%s/WLR.dat_%s" % (config.OUTPUT_DIR, suffix),
                   w, fmt="%.4f")

    def getInfo(self):
        return self.model
# class SLRModel:
#
#     def __init__(self):
#         self.isFitAndPredict = True
#         self.name = "SLR"
#         self.repred = None
#     def getParams(self):
#         return "Default"
#     def fitAndPredict(self, inputTrain, outputTrain, inputTest):
#         import warnings
#         from sklearn.exceptions import ConvergenceWarning
#         from models.baselines.slr import SLR
#         warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
#
#         print(inputTrain.shape, outputTrain.shape, inputTest.shape)
#
#         nClass = outputTrain.shape[1]
#         outputs = []
#         reps = []
#         nTest = inputTest.shape[0]
#         print("SLR for %s classes" % nClass)
#         self.model = SLR()
#         for i in range(nClass):
#             #if i % 10 == 0:
#             print("\r%s" % i, end ="")
#             output = outputTrain[:, i]
#             ar = checkOneClass(output, nTest)
#             ar2 = checkOneClass(output, inputTrain.shape[0])
#
#             # print clf
#             if type(ar) == int:
#                 self.model.fit(inputTrain, output)
#                 output = self.model.predict(inputTest)
#                 rep = self.model.predict(inputTrain)
#             else:
#                 output = ar
#                 rep = ar2
#             outputs.append(output)
#             reps.append(rep)
#
#         outputs = np.vstack(outputs).transpose()
#         reps = np.vstack(reps).transpose()
#         # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
#         self.repred = reps
#
#         print(outputs.shape)
#         print("\nDone")
#         return outputs
#
#     def getInfo(self):
#         return self.model

class LogisticModel:

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "LR"
        self.repred = None
    def getParams(self):
        return "Default"
    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("LR for %s classes" % nClass)
        model = LogisticRegression(C=config.LR_C, solver='lbfgs')
        self.model = model
        for i in range(nClass):
            # if i % 10 == 0:
            #    print("\r%s" % i)
            output = outputTrain[:, i]
            ar = checkOneClass(output, nTest)
            ar2 = checkOneClass(output, inputTrain.shape[0])

            # print clf
            if type(ar) == int:
                model.fit(inputTrain, output)
                output = model.predict_proba(inputTest)[:, 1]
                rep = model.predict_proba(inputTrain)[:, 1]
            else:
                output = ar
                rep = ar2
            outputs.append(output)
            reps.append(rep)

        outputs = np.vstack(outputs).transpose()
        reps = np.vstack(reps).transpose()
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return self.model

#
# class ParallelLogisticModel:
#
#     def __init__(self):
#         self.isFitAndPredict = True
#         self.name = "PLR"
#         self.repred = None
#     def getParams(self):
#         return "Default"
#     def __call__(self, i):
#         return self.lrModel(i)
#
#     def lrModel(self, i):
#         import warnings
#         from sklearn.exceptions import ConvergenceWarning
#         warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
#
#         def checkOneClass(inp, nSize):
#             s = sum(inp)
#             if s == 0:
#                 ar = np.zeros(nSize)
#             elif s == len(inp):
#                 ar = np.ones(nSize)
#             else:
#                 ar = -1
#             return ar
#
#         model2 = LogisticRegression(C=config.LR_C, solver='lbfgs')
#         output = self.sharedOutputTrain.array[:, i]
#         nTest = self.sharedInputTest.array.shape[0]
#         ar = checkOneClass(output, nTest)
#         ar2 = checkOneClass(output, self.sharedInputTrain.array.shape[0])
#
#         if type(ar) == int:
#
#             model2.fit(self.sharedInputTrain.array, output)
#             output = model2.predict_proba(self.sharedInputTest.array)[:, 1]
#             rep = model2.predict_proba(self.sharedInputTrain.array)[:, 1]
#         else:
#             output = ar
#             rep = ar2
#         # print ("X")
#         return output, rep, i
#
#     def fitAndPredict(self, inputTrain, outputTrain, inputTest):
#
#         print(inputTrain.shape, outputTrain.shape, inputTest.shape)
#         nClass = outputTrain.shape[1]
#         self.model = "Parallel LR"
#         print("PLR for %s classes" % nClass)
#
#         from models.baselines.sharedMem import SharedNDArray
#         from multiprocessing import Pool
#
#         self.sharedInputTrain = SharedNDArray.copy(inputTrain)
#         self.sharedInputTest = SharedNDArray.copy(inputTest)
#         self.sharedOutputTrain = SharedNDArray.copy(outputTrain)
#
#         print("In parallel mode")
#         # start = time.time()
#
#         iters = np.arange(0, nClass)
#         pool = Pool(config.N_PARALLEL)
#
#         adrOutputs = pool.map_async(self, iters)
#
#         pool.close()
#         pool.join()
#
#         while not adrOutputs.ready():
#             print("num left: {}".format(adrOutputs._number_left))
#             sys.stderr.flush()
#             time.sleep(.1)
#             #adrOutputs.wait(1)
#         # pool.close()
#         # pool.join()
#         outputs = []
#         reps = []
#         print(adrOutputs)
#         # print(adrOutputs.get())
#         dout = dict()
#         for output in adrOutputs.get():
#             dout[output[2]] = output[0], output[1]
#
#         for ii in range(len(dout)):
#             out1, out2 = dout[ii]
#             outputs.append(out1)
#             reps.append(out2)
#
#         # end = time.time()
#         # print("Elapsed: ", end - start)
#         outputs = np.vstack(outputs).transpose()
#         reps = np.vstack(reps).transpose()
#         self.repred = reps
#
#         print(outputs.shape)
#         print("\nDone")
#         return outputs
#
#     def getInfo(self):
#
#         return self.model


class MFModel:
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "MF"
    def getParams(self):
        return "Default"
    def fitAndPredict(self, intpuTrain, outputTrain, inputTest):
        from sklearn.decomposition import NMF

        self.model = NMF(config.N_FEATURE)
        chemFeatures = self.model.fit_transform(outputTrain)
        adrFeatures = self.model.components_

        nTrain, nTest = intpuTrain.shape[0], inputTest.shape[0]
        outSize = outputTrain.shape[1]
        simMatrix = np.ndarray((nTest, nTrain), dtype=float)
        for i in range(nTest):
            for j in range(nTrain):
                simMatrix[i][j] = utils.getTanimotoScore(inputTest[i], intpuTrain[j])

        args = np.argsort(simMatrix, axis=1)[:, ::-1]
        args = args[:, :config.KNN]
        # print args

        testFeatures = []
        for i in range(nTest):
            newF = np.zeros(config.N_FEATURE, dtype=float)
            matches = args[i]
            simScores = simMatrix[i, matches]
            ic = -1
            sum = 1e-10
            for j in matches:
                ic += 1
                newF += simScores[ic] * chemFeatures[j]
                sum += simScores[ic]
            newF /= sum
            testFeatures.append(newF)
        testVecs = np.vstack(testFeatures)
        self.repred = np.matmul(chemFeatures, adrFeatures)
        out = np.matmul(testVecs, adrFeatures)
        return out

    def getInfo(self):
        return self.model


class NeuSK:
    def __init__(self):
        self.isFitAndPredict = True
        self.name = "NeuSK"
    def getParams(self):
        return "Default"
    def fitAndPredict(self, input, output, inputtest):
        self.model = MLPRegressor((config.HIDDEN_1, config.HIDDEN_2), activation='relu')
        self.model.fit(input, output)
        self.repred = self.model.predict(input)
        return self.model.predict(inputtest)

    def getInfo(self):
        return self.model


class MultiSVM:

    def __init__(self):
        self.isFitAndPredict = True
        self.name = "SVM"
    def getParams(self):
        return "Default"
    def svmModel(self, i):

        # print (os.getpid(), i)
        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        model2 = svm.SVC(C=config.SVM_C, gamma='auto', kernel='rbf', probability=True)
        output = self.sharedOutputTrain.array[:, i]
        nTest = self.sharedInputTest.array.shape[0]
        ar = checkOneClass(output, nTest)
        ar2 = checkOneClass(output, self.sharedInputTrain.array.shape[0])

        if type(ar) == int:

            model2.fit(self.sharedInputTrain.array, output)
            output = model2.predict_proba(self.sharedInputTest.array)[:, 1]
            rep = model2.predict_proba(self.sharedInputTrain.array)[:, 1]
        else:
            output = ar
            rep = ar2

        return output, rep, i

    def __call__(self, i):
        return self.svmModel(i)

    def fitAndPredict(self, inputTrain, outputTrain, inputTest):
        from sklearn import svm

        print(inputTrain.shape, outputTrain.shape, inputTest.shape)

        def checkOneClass(inp, nSize):
            s = sum(inp)
            if s == 0:
                ar = np.zeros(nSize)
            elif s == len(inp):
                ar = np.ones(nSize)
            else:
                ar = -1
            return ar

        nClass = outputTrain.shape[1]
        outputs = []
        reps = []
        nTest = inputTest.shape[0]
        print("SVM for %s classes" % nClass)
        model = svm.SVC(C=config.SVM_C, gamma='auto', kernel='rbf', probability=True)
        self.model = model

        from utils.sharedMem import SharedNDArray
        from multiprocessing import Pool

        self.sharedInputTrain = SharedNDArray.copy(inputTrain)
        self.sharedInputTest = SharedNDArray.copy(inputTest)
        self.sharedOutputTrain = SharedNDArray.copy(outputTrain)

        if config.SVM_PARALLEL:
            print("In parallel mode")
            start = time.time()

            iters = np.arange(0, nClass)
            pool = Pool(config.N_PARALLEL)
            adrOutputs = pool.map_async(self, iters)
            pool.close()
            pool.join()

            outputs = []
            reps = []
            while not adrOutputs.ready():
                print("num left: {}".format(adrOutputs._number_left))
                adrOutputs.wait(1)

            print(adrOutputs)
            dout = dict()
            for output in adrOutputs.get():
                dout[output[2]] = output[0], output[1]

            for ii in range(len(dout)):
                out1, out2 = dout[ii]
                outputs.append(out1)
                reps.append(out2)

            end = time.time()
            print("Elapsed: ", end - start)

        else:
            print("In sequential mode")
            for i in range(nClass):
                if i % 10 == 0:
                    print("\r%s" % i, end="")

                output = outputTrain[:, i]
                ar = checkOneClass(output, nTest)
                ar2 = checkOneClass(output, inputTrain.shape[0])
                # print(inputTrain.shape, outputTrain.shape, output.shape)

                # print clf
                if type(ar) == int:
                    model.fit(inputTrain, output)
                    output = model.predict_proba(inputTest)[:, 1]
                    rep = model.predict_proba(inputTrain)[:, 1]
                else:
                    output = ar
                    rep = ar2
                outputs.append(output)
                reps.append(rep)

        outputs = np.vstack(outputs).transpose()
        reps = np.vstack(reps).transpose()
        # print("In Train: ", roc_auc_score(outputTrain.reshape(-1), reps.reshape(-1)))
        self.repred = reps

        print(outputs.shape)
        print("\nDone")
        return outputs

    def getInfo(self):
        return self.model
