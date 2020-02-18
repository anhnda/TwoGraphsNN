import numpy as np
import config
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
from numpy.linalg import pinv

scca = importr('scca')


class SCCAR:
    def __init__(self):
        self.isFitAndPredict = "True"
        self.name = "SCCAR"
        self.model = "SCCAR"

    def fit(self, input, output):
        numpy2ri.activate()
        sccaModel = scca.scca(input, output, nc=config.N_SCCA, center=False)
        self.WX = np.asmatrix(sccaModel.rx2('A'))
        self.WY = np.asmatrix(sccaModel.rx2('B'))
        numpy2ri.deactivate()

    def predict(self, input):
        xwx = np.matmul(input, self.WX)
        xwxwyt = np.matmul(xwx, self.WY.transpose())
        yyt = np.matmul(self.WY, self.WY.transpose())
        invyyt = pinv(yyt)
        pred = np.matmul(xwxwyt, invyyt)
        return np.asarray(pred)

    def fitAndPredict(self, input, output, inputTest):
        self.fit(input, output)
        self.repred = self.predict(input)
        pred = self.predict(inputTest)
        print(self.repred.shape, pred.shape)
        return pred

    def getParams(self):
        return "SCCA: %s " % config.N_SCCA
