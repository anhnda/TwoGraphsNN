import torch
import numpy as np

import itertools
import random

v1 = np.arange(0,3)
v2 = np.arange(4,8)
v = list(itertools.product(v1, v2))
xx = np.asarray(v).transpose()
print (xx)
