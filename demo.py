import torch

import numpy as np


def getT(m1, m2, v1, v2, n1, n2):
    return (m1-m2) / (v1* v1/ n1 + v2 * v2 / n2)

print (getT(0.7654, 0.0037, 0.7612, 0.0036, 10, 10))
