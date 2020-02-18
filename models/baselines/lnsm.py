import numpy as np
import utils, config
from numpy.linalg import pinv
from qpsolvers import solve_qp


def getRowLNSM(v, mInp, idx=-1):
    nObj = mInp.shape[0]
    ar = np.zeros(nObj)
    for i, inp in enumerate(mInp):
        ar[i] = utils.utils.getTanimotoScore(v, inp)
    if idx >= 0:
        ar[idx] = -10
    args = np.argsort(ar)[::-1][:config.CS_KNN]
    P = np.ndarray((config.CS_KNN, config.CS_KNN))
    for i in range(config.CS_KNN):
        for j in range(i, config.CS_KNN):
            P[i][j] = np.dot(v - mInp[args[i]], v - mInp[args[j]])
            P[j][i] = P[i][j]

    I = np.diag(np.ones(config.CS_KNN))
    P = P + I
    q = np.zeros(config.CS_KNN)
    gg = np.ndarray(config.CS_KNN)
    gg.fill(-1)
    G = np.diag(gg)
    h = np.zeros(config.CS_KNN)
    b = np.ones(1)
    A = np.ones(config.CS_KNN)
    re = solve_qp(P, q, G, h, A, b)
    out = np.zeros(nObj)
    for i in range(config.CS_KNN):
        out[args[i]] = re[i]
    return out


def learnLNSM(mInp, mOut):
    nObj = mInp.shape[0]
    simAr = []
    for i in range(nObj):
        lnsm = getRowLNSM(mInp[i], mInp, i)
        simAr.append(lnsm)
    W = np.vstack(simAr)

    I = np.diag(np.ones(nObj))
    W = W * config.ALPHA
    I = I - W
    Y = pinv(I)
    Y *= 1 - config.ALPHA
    Y = np.matmul(Y, mOut)
    return Y
