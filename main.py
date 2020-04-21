from evaluator import Evaluator
from optparse import OptionParser
import config


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



def runLevel1GNN():
    from models.trainLevel1GNN import WrapperLevel1GNN
    model = WrapperLevel1GNN()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runLevel2GNN():
    from models.trainLevel2GNN import WrapperLevel2GNN
    model = WrapperLevel2GNN()
    evaluator = Evaluator()
    evaluator.evalModel(model)


def runG3N2LevelV1():
    from models.trainG3N2Level import WrapperG3N2Level
    model = WrapperG3N2Level()
    evaluator = Evaluator()
    evaluator.evalModel(model)




if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="G3N1",
                      help="MODELNAME:\n"
                           "NeuN: neural feedforward networks,\n")

    parser.add_option("-i", "--fold", dest="ifold", type='int', default=config.IFOLD)

    (options, args) = parser.parse_args()
    parseConfig(options)
    modelName = options.modelName

    if modelName == "L1":
        runLevel1GNN()
    elif modelName == "L2":
        runLevel2GNN()
    elif modelName == "G3N1":
        runG3N2LevelV1()

    else:

        print("Method %s is not implemented" % modelName)
        exit(-1)
