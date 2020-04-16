import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
# from models.gnn2.EAT import EATConv
from models.g3n.XGAT import XGAT
from models.g3n.XSAGE import XSAGE
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import math


class Level2GNN(torch.nn.Module):
    def __init__(self, outSize, layerType=SAGEConv, maxNode=10000):
        super(Level2GNN, self).__init__()

        self.LAYER_TYPE = layerType
        self.LAYERS = []

        N = 5
        for i in range(N):
            layer = self.LAYER_TYPE(config.EMBED_DIM, config.EMBED_DIM)
            self.LAYERS.append(layer)

        self.act = F.relu

        self.linear1 = Linear(config.EMBED_DIM, config.EMBED_DIM)
        self.linear2 = Linear(config.EMBED_DIM, outSize)

        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=maxNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)


        # self.linear1.weight.data.uniform_(0.001, 0.3)
        self.linear2.weight.data.uniform_(0.001, 0.3)

        # Molecule graph neural net

    def my_reset_params(self, tensor, size=10):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(0.0, bound)

    def forward(self, x, edges):

        x = self.nodesEmbedding(x)
        x = x.squeeze()

        for i in range(config.N_LAYER_LEVEL_2):
            x = self.LAYERS[i](x, edges)
            x = F.relu(x)
        return x

    def calOut(self, x, keyIds):
        o = x[keyIds]
        # o = self.linear1(o)
        # o = F.relu(o)
        o = self.linear2(o)
        o2 = F.relu(o)
        return o2
