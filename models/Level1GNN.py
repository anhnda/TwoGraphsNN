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


class Level1GNN(torch.nn.Module):
    def __init__(self, outSize, layerType=GATConv, numNode=10000 ):
        super(Level1GNN, self).__init__()

        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)
        self.CONV_LAYER = layerType
        # Molecule graph neural net

        self.convs = []
        self.pools = []

        for i in range(config.N_LAYER_LEVEL_1):
            self.convs.append(self.CONV_LAYER(config.EMBED_DIM, config.EMBED_DIM))
            self.pools.append(TopKPooling(config.EMBED_DIM, ratio=0.8))

        self.lin1 = torch.nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.lin2 = torch.nn.Linear(config.EMBED_DIM, config.EMBED_DIM)

        self.lout1 = Linear(config.EMBED_DIM, config.EMBED_DIM)
        self.lout2 = Linear(config.EMBED_DIM, outSize)

    def my_reset_params(self, tensor, size=10):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(0.0, bound)

    def forward(self, graphBatch):

        xAtom, edge_index, batch = graphBatch.x, graphBatch.edge_index, graphBatch.batch
        xAtom = self.nodesEmbedding(xAtom)
        xAtom = xAtom.squeeze(1)

        xs = []
        for i in range(config.N_LAYER_LEVEL_1):
            xAtom = F.relu(self.convs[i](xAtom, edge_index))
            # xAtom, edge_index, _, batch, _, _ = self.pools[i](xAtom, edge_index, None, batch)
            xi = torch.cat([gmp(xAtom, batch), gap(xAtom, batch)], dim=1)
            xs.append(xi)

        xDrug = xs[-1]

        xDrug = self.lin1(xDrug)
        xDrug = F.relu(xDrug)

        return xDrug

    def calOut(self, x, keyIds):
        o = x[keyIds]
        # o = self.linear1(o)
        # o = F.relu(o)
        o = self.lout2(o)
        # o = F.relu(o)
        return o
