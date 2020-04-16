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





def runLevel2GNN():
    from models.trainLevel2GNN import WrapperLevel2GNN
    model = WrapperLevel2GNN()
    evaluator = Evaluator()
    evaluator.evalModel(model)








if __name__ == "__main__":
    parser = OptionParser()

    parser.add_option("-m", "--model", dest="modelName", type='string', default="L2",
                      help="MODELNAME:\n"
                           "NeuN: neural feedforward networks,\n")

    parser.add_option("-i", "--fold", dest="ifold", type='int', default=config.IFOLD)

    (options, args) = parser.parse_args()
    parseConfig(options)
    modelName = options.modelName

    if modelName == "L2":
        runLevel2GNN()
    else:

        print("Method %s is not implemented" % modelName)
        exit(-1)
