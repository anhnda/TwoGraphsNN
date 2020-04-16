import torch
import numpy as np

import itertools
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
nodeVecs1 = torch.from_numpy(np.zeros((2, 2), dtype=float)).float()
nodeVecs2 = torch.from_numpy(np.zeros((2, 4), dtype=float)).float()
v = [nodeVecs1]
v.append(nodeVecs2)
ar = torch.cat(v, dim=1)


print(ar)
