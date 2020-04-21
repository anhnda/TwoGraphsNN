import torch
import numpy as np

import itertools
import random
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv

def setEmbedding(self, embedding1, embedding2, size2):
    nDim = embedding2.shape[1]


embedding1 = torch.nn.Embedding(20, 50)
embedding2 = torch.nn.Embedding(20, 50)

print(embedding1.weight.data)

