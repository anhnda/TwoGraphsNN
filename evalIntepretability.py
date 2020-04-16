import config
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.logger.logger2 import MyLogger

from dataFactory.loader4 import BioLoader4

import numpy as np

from utils import utils
import torch
from dataFactory.loader4 import BioLoader4


def evalPreRec(top, predictedList, targetSet):
    prec = np.zeros(top, dtype=float)
    rec = np.zeros(top, dtype=float)
    t = min(top, len(predictedList))
    for i in range(t):
        v = predictedList[i]
        if v in targetSet:
            prec[i:] += 1
            rec[i:] += 1
    for i in range(top):
        prec[i] /= (i + 1)
        rec[i] /= len(targetSet)
    return prec, rec


def trainAllMPNN(model, bioLoader):
    model.fit(bioLoader)
    path = "%s/%s" % (config.SAVEMODEL_DIR, model.name)
    torch.save(model.model, path)


def executeMPNN433():
    from models.MPNNX4_33 import MPNNX4_33
    bioloader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioloader4.createTrainTestVal(trainPath, allTrain=True)
    model = MPNNX4_33()

    trainAllMPNN(model, bioloader4)


def evalTopMPNN433():
    from models.net4_3 import Net4_3

    bioLoader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioLoader4.createTrainTestVal(trainPath, allTrain=True)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net4 = Net4_3(numNode=config.MAX_NODE + 1, numAtomFeature=bioLoader4.N_ATOMFEATURE)
    # model = net4.to(device)
    model = torch.load("%s/%s" % (config.SAVEMODEL_DIR, "MPNNX433"))

    def createId2RowId(idList):
        d = dict()
        drugIdSet = set()

        for idx in idList:
            d[idx] = len(d)
            if idx not in drugIdSet:
                drugIdSet.add(idx)
            else:
                print("Duplicate : ", idx)
        return d
    dDrugId2RowId = createId2RowId(bioLoader4.drugTrainIdList)

    inputTrain = bioLoader4.trainInpMat[:,
                 config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader4.nProtein]
    outputTrain = bioLoader4.trainOutMatrix.numpy()

    def evalAPair(drugId, seId):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        drugE, seE, x = model.forward(bioLoader4.x,
                                      bioLoader4.drugGraphData.edge_index, bioLoader4.seGraphData.edge_index,
                                      bioLoader4.drugTrainNodeIds,
                                      bioLoader4.seNodeIds,
                                      bioLoader4.graphBatch,
                                      bioLoader4.nDrug
                                      )
        out = torch.matmul(drugE[drugId], seE[seId])
        out.backward()
        grad = model.nodesEmbedding.weight.grad

        # grad = torch.mul(grad, model.nodesEmbedding.weight)
        grad = torch.mul(grad, grad)
        scores = torch.sum(grad, dim=1)
        scores = scores[bioLoader4.PROTEIN_OFFSET: bioLoader4.PROTEIN_OFFSET + bioLoader4.nProtein]
        scores = scores.detach().numpy()
        scores = np.abs(scores)
        scores = np.multiply(scores, inputTrain[drugId, : ])

        sortedIndices = np.argsort(scores)[::-1]


        # Print interacted proteins:
        nonZeros = np.nonzero(inputTrain[drugId, :])[0]
        for ix in nonZeros:
            print(bioLoader4.id2ProteinName[ix], ", ", end="")
        print()
        # Print scores with names:
        for ix in sortedIndices[:10]:
            print(scores[ix], bioLoader4.id2ProteinName[ix], ", ", end="")
        print ()






        return sortedIndices


    err = 0
    numValid = 0
    N_TOP = 10
    pre = np.zeros(N_TOP)
    rec = np.zeros(N_TOP)



    for pair, proteinDart in bioLoader4.dartBenchMark.items():

        drugId, seId = pair
        print(drugId, bioLoader4.drugId2Inchikey[drugId], bioLoader4.drugInchi2Name[bioLoader4.drugId2Inchikey[drugId]])
        print(seId, bioLoader4.dseId2Names[seId])

        drugId = dDrugId2RowId[drugId]
        if outputTrain[drugId, seId] < 1:
            err += 1
            print("Err in data")
            continue
        numValid += 1
        sortedIndices = evalAPair(drugId, seId)
        preci, reci = evalPreRec(N_TOP, sortedIndices, proteinDart)
        pre += preci
        rec += reci
        print("Eval: ", sortedIndices, "#", proteinDart)
        print(preci, reci)
    pre /= numValid
    rec /= numValid
    print(err, numValid)
    print(pre, rec)


def trainAllSLR(model, bioLoader):
    inputTrain = bioLoader.trainInpMat
    outputTrain = bioLoader.trainOutMatrix.numpy()
    print (inputTrain.shape, outputTrain.shape)
    model.fit(inputTrain, outputTrain)
    path = "%s/%s" % (config.SAVEMODEL_DIR, "SLR")
    np.savetxt(path, model.getW())


def executeSLR():
    from models.baselines.slr import PSLR
    bioloader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioloader4.createTrainTestVal(trainPath, allTrain=True)
    model = PSLR()
    trainAllSLR(model, bioloader4)




def executeSCCA():
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    bioloader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioloader4.createTrainTestVal(trainPath, allTrain=True)

    inputTrain = bioloader4.trainInpMat[:,
                 config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioloader4.nProtein]
    outputTrain = bioloader4.trainOutMatrix.numpy()
    scca = importr('scca')


    numpy2ri.activate()
    sccaModel = scca.scca(inputTrain, outputTrain, nc=config.N_SCCA, center=False)
    WX = np.asmatrix(sccaModel.rx2('A'))
    WY = np.asmatrix(sccaModel.rx2('B'))
    UX = np.asmatrix(sccaModel.rx2('U'))
    VY = np.asmatrix(sccaModel.rx2('V'))
    numpy2ri.deactivate()
    np.savetxt("%s/scca/WX" % config.SAVEMODEL_DIR, WX)
    np.savetxt("%s/scca/WY" % config.SAVEMODEL_DIR, WY)
    np.savetxt("%s/scca/UX" % config.SAVEMODEL_DIR, UX)
    np.savetxt("%s/scca/VY" % config.SAVEMODEL_DIR, VY)


def evalTopSLR():
    from models.baselines.slr import PSLR
    bioLoader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioLoader4.createTrainTestVal(trainPath, allTrain=True)
    path = "%s/%s" % (config.SAVEMODEL_DIR, "SLR")
    weights = np.loadtxt(path)
    weights = np.transpose(weights)
    print(weights.shape)
    weights = weights[:, config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader4.nProtein]

    def createId2RowId(idList):
        d = dict()
        drugIdSet = set()

        for idx in idList:
            d[idx] = len(d)
            if idx not in drugIdSet:
                drugIdSet.add(idx)
            else:
                print("Duplicate : ", idx)
        return d
    dDrugId2RowId = createId2RowId(bioLoader4.drugTrainIdList)

    inputTrain = bioLoader4.trainInpMat[:,
                 config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader4.nProtein]
    outputTrain = bioLoader4.trainOutMatrix.numpy()
    def evalAPair(drugId, seId):
        seFeatureWeights = weights[seId]
        matchingScores = seFeatureWeights
        matchingScores = np.multiply(inputTrain[drugId, :], matchingScores)
        # matchingScores = np.abs(matchingScores)
        sortedIndices = np.argsort(matchingScores)[::-1]
        print(matchingScores[sortedIndices])
        return sortedIndices

    evalAPair(0, 0)

    err = 0
    numValid = 0
    N_TOP = 10
    pre = np.zeros(N_TOP)
    rec = np.zeros(N_TOP)

    for pair, proteinDart in bioLoader4.dartBenchMark.items():
        drugId, seId = pair
        drugId = dDrugId2RowId[drugId]

        if outputTrain[drugId, seId] < 1:
            err += 1
            print("Err in data")
            continue
        numValid += 1
        sortedIndices = evalAPair(drugId, seId)
        preci, reci = evalPreRec(N_TOP, sortedIndices, proteinDart)
        pre += preci
        rec += reci
        print("Eval: ", sortedIndices, "#", proteinDart)
        print(preci, reci)
    pre /= numValid
    rec /= numValid
    print(err, numValid)
    print(pre, rec)

def evalTopSCCA():

    bioLoader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioLoader4.createTrainTestVal(trainPath, allTrain=True)

    WX = np.loadtxt("%s/scca/WX" % config.SAVEMODEL_DIR)
    WY = np.loadtxt("%s/scca/WY" % config.SAVEMODEL_DIR)
    UX = np.loadtxt("%s/scca/UX" % config.SAVEMODEL_DIR)
    VY = np.loadtxt("%s/scca/VY" % config.SAVEMODEL_DIR)




    def createId2RowId(idList):
        d = dict()
        drugIdSet = set()

        for idx in idList:
            d[idx] = len(d)
            if idx not in drugIdSet:
                drugIdSet.add(idx)
            else:
                print("Duplicate : ", idx)
        return d
    dDrugId2RowId = createId2RowId(bioLoader4.drugTrainIdList)

    inputTrain = bioLoader4.trainInpMat[:,
                 config.CHEM_FINGERPRINT_SIZE: config.CHEM_FINGERPRINT_SIZE + bioLoader4.nProtein]
    outputTrain = bioLoader4.trainOutMatrix.numpy()
    def evalAPair(drugId, seId):
        ux = UX[drugId, :]
        vy = VY[seId, :]
        cScores = np.multiply(ux, vy)
        cScores[cScores<0] = 0
        print (cScores)

        sePattern = np.zeros(bioLoader4.nSe)
        sePattern[seId] = 1
        seMatching = np.dot(sePattern, WY)
        seMatching[seMatching < 0] = 0

        cScores2 = np.multiply(cScores, seMatching)


        matchingScores = np.dot(WX, cScores2)

        matchingScores = np.multiply(inputTrain[drugId, :], matchingScores)
        sortedIndices = np.argsort(matchingScores)[::-1]
        print(matchingScores[sortedIndices])
        return sortedIndices


    # evalAPair(0, 0)

    err = 0
    numValid = 0
    N_TOP = 10
    pre = np.zeros(N_TOP)
    rec = np.zeros(N_TOP)

    for pair, proteinDart in bioLoader4.dartBenchMark.items():
        drugId, seId = pair
        drugId = dDrugId2RowId[drugId]

        if outputTrain[drugId, seId] < 1:
            err += 1
            print("Err in data")
            continue
        numValid += 1
        sortedIndices = evalAPair(drugId, seId)
        preci, reci = evalPreRec(N_TOP, sortedIndices, proteinDart)
        pre += preci
        rec += reci
        print("Eval: ", sortedIndices, "#", proteinDart)
        print(preci, reci)
    pre /= numValid
    rec /= numValid
    print(err, numValid)
    print(pre, rec)


def evalData():
    bioLoader4 = BioLoader4()
    trainPath = BioLoader4.getPathIFold(config.IFOLD)
    bioLoader4.createTrainTestVal(trainPath, allTrain=True)
    inputTrain = bioLoader4.trainInpMat
    outputTrain = bioLoader4.trainOutMatrix.numpy()
    inputTest = bioLoader4.testInpMat
    outputTest = bioLoader4.testOutMatrix.numpy()

    print(outputTrain.shape)
    print ("NATOMFE", bioLoader4.N_ATOMFEATURE)

    def createId2RowId(idList):
        d = dict()
        drugIdSet = set()

        for idx in idList:
            d[idx] = len(d)
            if idx not in drugIdSet:
                drugIdSet.add(idx)
            else:
                print("Duplicate : ", idx)
        return d

    d = createId2RowId(bioLoader4.drugTrainIdList)
    print(len(d))

    for pair, proteinDart in bioLoader4.dartBenchMark.items():
        drugId, seId = pair
        drugId = d[drugId]
        if outputTrain[drugId, seId] < 1:
            print("Err in data", drugId, seId)
            continue




if __name__ == "__main__":
    # executeMPNN433()
    evalTopMPNN433()
    # evalData()
    # executeSLR()
    # evalTopSLR()
    # executeSCCA()
    # evalTopSCCA()
