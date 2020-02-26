import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear


class Net3(torch.nn.Module):
    def __init__(self, numNode=10000):
        super(Net3, self).__init__()

        self.convD1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convD2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)

        self.convS1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convS2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)

        self.L1 = Linear(config.CHEM_FINGERPRINT_SIZE, config.EMBED_DIM * 2)
        self.actL1 = F.relu
        self.L2 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.actL2 = F.relu

        self.linear1 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.act1 = F.relu
        self.linear2 = Linear(config.EMBED_DIM, 1)
        self.act2 = F.relu

        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)

    def forward(self, x, drugEdges, seEdges, drugNodes, seNodes, drugFeatures):

        nDrug = drugFeatures.shape[0]
        x = self.nodesEmbedding(x[nDrug:])
        x = x.squeeze(1)

        xDrug = self.L1(drugFeatures)
        xDrug = self.actL1(xDrug)
        xDrug = self.L2(xDrug)
        xDrug = self.actL2(xDrug)

        x = torch.cat((xDrug, x), dim=0)

        # Conv Drug:
        x = self.convD1(x, drugEdges)
        x = F.relu(x)
        x = self.convD2(x, drugEdges)
        x = F.relu(x)
        # Conv SE:
        x = self.convS1(x, seEdges)
        x = F.relu(x)
        x = self.convS2(x, seEdges)
        x = F.relu(x)

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
