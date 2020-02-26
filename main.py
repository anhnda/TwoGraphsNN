from evaluator import Evaluator
from optparse import OptionParser
import config

#
# def runNeuModel():
#     model = NeuNModel()
#     evaluator = Evaluator()
#     evaluator.evalModel(model)


def runLR():
    from models.baselines.omodels import LogisticModel
    model = LogisticModel()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runPSLR():
    from models.baselines.omodels import PSLRModel
    model = PSLRModel()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runLNSM():
    from models.baselines.omodels import LNSM
    model = LNSM()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMF():
    from models.baselines.omodels import MFModel
    model = MFModel()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runSCCAR():
    from models.baselines.sccar import SCCAR
    model = SCCAR()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runNeuSK():
    from models.baselines.omodels import NeuSK
    model = NeuSK()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runSVM():
    from models.baselines.omodels import MultiSVM
    model = MultiSVM()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runCSMF():
    from models.baselines.CSMF import CSMF
    model = CSMF()
    evaluator = Evaluator()
    evaluator.evalModel(model)



def runMPNN():
    from models.MPNNX import MPNNX
    model = MPNNX()
    evaluator = Evaluator()
    evaluator.evalModel(model)

def runMPNN3():
    from models.MPNNX3 import MPNNX3
    model = MPNNX3()
    evaluator = Evaluator()
    evaluator.evalModel(model)



def runMPNN4():
    from models.MPNNX4 import MPNNX4
    model = MPNNX4()
    evaluator = Evaluator()
    evaluator.evalModel(model)

def convertStringToBoolean(val):
    print(val)
    if type(val) == bool:
        return val
    if val.upper() == "TRUE":
        return True
    elif val.upper() == "FALSE":
        return False
    else:
        print("Fatal error: Wrong Boolean String")
        exit(-1)


def parseConfig(options):
    config.IFOLD = options.ifold




if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="MPNN4",
                      help="MODELNAME:\n"
                           "NeuN: neural feedforward networks,\n")

    parser.add_option("-i", "--fold", dest="ifold", type='int', default=config.IFOLD)




    (options, args) = parser.parse_args()
    parseConfig(options)
    modelName = options.modelName

    if modelName == "NeuSK":
        runNeuSK()
    elif modelName == "LR":
        runLR()
    elif modelName == "PSLR":
        runPSLR()
    elif modelName == "MF":
        runMF()
    elif modelName == "SVM":
        runSVM()
    elif modelName == "LNSM":
        runLNSM()
    elif modelName == "CSMF":
        runCSMF()
    elif modelName == "SCCA":
        runSCCAR()
    elif modelName == "MPNN":
        runMPNN()
    elif modelName == "MPNN3":
        runMPNN3()
    elif modelName == "MPNN4":
        runMPNN4()
    else:
        print("Method %s is not implemented" % modelName)
        exit(-1)
