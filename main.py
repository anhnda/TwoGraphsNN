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


def runMPNN1():
    from models.MPNN1 import MPNN1
    model = MPNN1()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX1():
    from models.MPNNX1 import MPNNX1
    model = MPNNX1()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX3():
    from models.MPNNX3 import MPNNX3
    model = MPNNX3()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX32():
    from models.MPNNX3_2 import MPNNX3_2
    model = MPNNX3_2()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX4():
    from models.MPNNX4 import MPNNX4
    model = MPNNX4()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX5():
    from models.MPNNX5 import MPNNX5
    model = MPNNX5()
    evaluator = Evaluator()
    evaluator.evalModel(model)

def runMPNNX52():
    from models.MPNNX52 import MPNNX52
    model = MPNNX52()
    evaluator = Evaluator()
    evaluator.evalModel(model)

def runMPNNX6():
    from models.MPNNX6 import MPNNX6
    model = MPNNX6()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX42():
    from models.MPNNX4_2 import MPNNX4_2
    model = MPNNX4_2()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNXP42():
    from models.MPNNX4P2 import MPNNXP4
    model = MPNNXP4()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNXP52():
    from models.MPNNXP5 import MPNNXP5
    model = MPNNXP5()
    evaluator = Evaluator()
    evaluator.evalModel(model)

def runMPNNX43():
    from models.MPNNX4_3 import MPNNX4_3
    model = MPNNX4_3()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX433():
    from models.MPNNX4_33 import MPNNX4_33
    model = MPNNX4_33()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runMPNNX44():
    from models.MPNNX4_4 import MPNNX4_4
    model = MPNNX4_4()
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

    parser.add_option("-m", "--model", dest="modelName", type='string', default="MPNNXP52",
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

    elif modelName == "MPNN1":
        runMPNN1()

    elif modelName == "MPNNX1":
        runMPNNX1()
    elif modelName == "MPNNX3":
        runMPNNX3()
    elif modelName == "MPNNX32":
        runMPNNX32()
    elif modelName == "MPNNX4":
        runMPNNX4()
    elif modelName == "MPNNX5":
        runMPNNX5()
    elif modelName == "MPNNX52":
        runMPNNX52()
    elif modelName == "MPNNX6":
        runMPNNX6()
    elif modelName == "MPNNX42":
        runMPNNX42()
    elif modelName == "MPNNXP42":
        runMPNNXP42()
    elif modelName == "MPNNXP52":
        runMPNNXP52()
    elif modelName == "MPNNX43":
        runMPNNX43()
    elif modelName == "MPNNX433":
        runMPNNX433()
    elif modelName == "MPNNX44":
        runMPNNX44()
    else:
        print("Method %s is not implemented" % modelName)
        exit(-1)
