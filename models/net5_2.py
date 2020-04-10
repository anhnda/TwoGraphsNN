import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.gnn.EAT import EATConv
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear

from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

import math


class Net52(torch.nn.Module):
    def __init__(self, numNode=10000, numAtomFeature=0):
        super(Net52, self).__init__()

        self.convD1 = GCNConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convD2 = GCNConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convD3 = GCNConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)

        # self.my_reset_params(self.convD1.weight, config.EMBED_DIM)
        # self.my_reset_params(self.convD1.bias, config.EMBED_DIM)
        #
        # self.my_reset_params(self.convD2.weight, config.EMBED_DIM)
        # self.my_reset_params(self.convD2.bias, config.EMBED_DIM)

        self.convS1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convS2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)

        # self.my_reset_params(self.convS1.weight, config.EMBED_DIM)
        # self.my_reset_params(self.convS1.bias, config.EMBED_DIM)
        #
        # self.my_reset_params(self.convS2.weight, config.EMBED_DIM)
        # self.my_reset_params(self.convS2.bias, config.EMBED_DIM)

        self.L1 = Linear(config.CHEM_FINGERPRINT_SIZE, config.EMBED_DIM * 2)
        self.L1C = Linear(config.CHEM_FINGERPRINT_SIZE + config.EMBED_DIM, config.EMBED_DIM * 2)
        self.actL1 = F.relu
        self.L2 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.actL2 = F.relu

        self.linear1 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.act1 = F.relu
        self.linear2 = Linear(config.EMBED_DIM, 1)
        self.act2 = F.relu

        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)

        # Molecule graph neural net

        self.mlinear1 = Linear(numAtomFeature, config.EMBED_DIM * 2)
        self.mact1 = F.relu
        self.mlinear2 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.mact2 = F.relu

        self.conv1 = EATConv(config.EMBED_DIM, config.EMBED_DIM, extProb=config.CROSS_PROB)
        self.conv1g = GATConv(config.EMBED_DIM, config.EMBED_DIM)

        self.pool1 = TopKPooling(config.EMBED_DIM, ratio=1)
        self.conv2 = EATConv(config.EMBED_DIM, config.EMBED_DIM, extProb=config.CROSS_PROB)
        self.conv2g = GATConv(config.EMBED_DIM, config.EMBED_DIM)

        self.pool2 = TopKPooling(config.EMBED_DIM, ratio=1)
        self.conv3 = EATConv(config.EMBED_DIM, config.EMBED_DIM, extProb=config.CROSS_PROB)
        self.conv3g = GATConv(config.EMBED_DIM, config.EMBED_DIM)

        self.pool3 = TopKPooling(config.EMBED_DIM, ratio=1)

        self.conv4 = EATConv(config.EMBED_DIM, config.EMBED_DIM, extProb=config.CROSS_PROB)
        self.conv4g = GATConv(config.EMBED_DIM, config.EMBED_DIM)
        self.pool4 = TopKPooling(config.EMBED_DIM, ratio=1)

        self.conv5 = EATConv(config.EMBED_DIM, config.EMBED_DIM, extProb=config.CROSS_PROB)
        self.conv5g = GATConv(config.EMBED_DIM, config.EMBED_DIM)
        self.pool5 = TopKPooling(config.EMBED_DIM, ratio=1)

        self.lin1 = torch.nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.lin2 = torch.nn.Linear(config.EMBED_DIM, config.EMBED_DIM)

        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()

    def my_reset_params(self, tensor, size=10):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(0.0, bound)

    def forward(self, x, drugEdges, seEdges, drugNodes, seNodes, proteinNodes, proteinWeight, drugGraphBatch, nDrug,
                drugFeatures=None):

        xAtomProtein, edge_index, batch = drugGraphBatch.x, drugGraphBatch.edge_index, drugGraphBatch.batch
        xAtomProtein = self.nodesEmbedding(xAtomProtein)
        xAtomProtein = xAtomProtein.squeeze(1)

        xAtomProtein = F.relu(self.conv1g(xAtomProtein, edge_index))
        xAtomProtein, edge_index, _, batch, _, _ = self.pool1(xAtomProtein, edge_index, None, batch)
        x1 = torch.cat([gmp(xAtomProtein, batch), gap(xAtomProtein, batch)], dim=1)

        xAtomProtein = F.relu(self.conv2g(xAtomProtein, edge_index))
        xAtomProtein, edge_index, _, batch, _, _ = self.pool2(xAtomProtein, edge_index, None, batch)
        x2 = torch.cat([gmp(xAtomProtein, batch), gap(xAtomProtein, batch)], dim=1)

        xAtomProtein = F.relu(self.conv3g(xAtomProtein, edge_index))
        xAtomProtein, edge_index, _, batch, _, _ = self.pool3(xAtomProtein, edge_index, None, batch)
        x3 = torch.cat([gmp(xAtomProtein, batch), gap(xAtomProtein, batch)], dim=1)

        xDrugProtein = x1 + x2 + x3
        xDrugProtein = self.lin1(xDrugProtein)
        xDrugProtein = self.act1(xDrugProtein)

        # xDrug = self.lin2(xDrug)
        # xDrug = self.act2(xDrug)

        nProtein = len(proteinNodes)
        x = self.nodesEmbedding(x[nDrug+nProtein:])

        x = x.squeeze(1)

        x = torch.cat((xDrugProtein, x), dim=0)

        # Conv Drug:
        x = self.convD1(x, drugEdges)
        x = F.relu(x)
        x = self.convD2(x, drugEdges)
        x = F.relu(x)

        # x = self.convD3(x, drugEdges)
        # x = F.relu(x)

        drugEmbedding = x[drugNodes]
        seEmbedding = x[seNodes]
        # re = torch.sigmoid(re)
        return drugEmbedding, seEmbedding, x

    def cal(self, drugE, seE):
        return torch.matmul(drugE, seE.t())

    def cal2(self, drugE, seE):
        nDrug, nDim = drugE.shape
        nSe, _ = seE.shape
        preRe = list()
        for i in range(nDrug):
            dE = drugE[i]
            dE = dE.squeeze()
            de = dE.expand((nSe, nDim))
            v = torch.cat((de, seE), dim=1)
            v = self.linear1(v)
            v = self.act1(v)
            v = self.linear2(v)
            # v = self.act2(v)
            v = v.squeeze()
            preRe.append(v)
        return torch.stack(preRe)
