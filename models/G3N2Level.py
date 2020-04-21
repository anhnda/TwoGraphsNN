import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.g3n.XSAGE import XSAGE
from models.g3n.XGAT import XGAT

from models.g3n.XGAT import XGAT
from models.g3n.XSAGE import XSAGE
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import math


class G3N2Level(torch.nn.Module):
    def __init__(self, outSize, layerType1=SAGEConv, layerType2=XSAGE, numNode=10000):
        super(G3N2Level, self).__init__()

        self.nodesEmbedding1 = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding1.weight.data.uniform_(0.001, 0.3)

        self.nodesEmbedding2 = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding2.weight.data.uniform_(0.001, 0.3)

        self.CONV_LAYER1 = layerType1
        self.convs1 = []
        self.pools1 = []

        self.CONV_LAYER2 = layerType2
        self.convs2 = []

        for i in range(config.N_LAYER_LEVEL_1):
            self.convs1.append(self.CONV_LAYER1(config.EMBED_DIM, config.EMBED_DIM))
            self.pools1.append(TopKPooling(config.EMBED_DIM, ratio=0.8))

        for i in range(config.N_LAYER_LEVEL_2):
            self.convs2.append(self.CONV_LAYER2(config.EMBED_DIM, config.EMBED_DIM))

        self.lin1 = torch.nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.lin2 = torch.nn.Linear(config.EMBED_DIM, config.EMBED_DIM)

        self.lout1 = Linear(config.EMBED_DIM, config.EMBED_DIM)
        self.lout2 = Linear(config.EMBED_DIM, outSize)

    def my_reset_params(self, tensor, size=10):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(0.0, bound)

    def forward(self, graphBatch1, graph2):

        # Graphs at level 1

        xAtom, edge_index, batch = graphBatch1.x, graphBatch1.edge_index, graphBatch1.batch
        xAtom = self.nodesEmbedding1(xAtom)
        xAtom = xAtom.squeeze(1)

        xs = []
        for i in range(config.N_LAYER_LEVEL_1):
            xAtom = F.relu(self.convs1[i](xAtom, edge_index))
            # xAtom, edge_index, _, batch, _, _ = self.pools[i](xAtom, edge_index, None, batch)
            xi = torch.cat([gmp(xAtom, batch), gap(xAtom, batch)], dim=1)
            xs.append(xi)

        xDrug = xs[-1]

        xDrug = self.lin1(xDrug)
        xDrug = F.relu(xDrug)

        # Initialize for level 2
        xDrugInit = xDrug.detach()
        self.setEmbedding(self.nodesEmbedding2, xDrugInit, xDrugInit.shape[0])

        # Update level 2
        x2, edges2 = graph2.x, graph2.edge_index

        x2 = self.nodesEmbedding2(x2)
        x2 = x2.squeeze()

        for i in range(config.N_LAYER_LEVEL_2):
            x2 = self.convs2[i](x2, edges2, xDrug)
            x2 = F.relu(x2)

        return x2

    def setEmbedding(self, embedding1, embedding2, size2):
        nDim = embedding2.shape[1]
        for i in range(size2):
            for j in range(nDim):
                embedding1.weight.data[i, j] = embedding2.data[i, j]

    def calOut(self, x, keyIds):
        o = x[keyIds]
        # o = self.linear1(o)
        # o = F.relu(o)
        o = self.lout2(o)
        # o = F.relu(o)
        return o
