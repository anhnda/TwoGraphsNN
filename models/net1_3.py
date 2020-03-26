import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv
from torch import sigmoid
import config

from torch.nn import Linear

class Net1(torch.nn.Module):
    def __init__(self, numNode=10000):
        super(Net1, self).__init__()

        self.conv1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.conv2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)

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


    def forward(self, x, edge_index, drugNodes, seNodes, drugFeatures):

        #nDrug = drugFeatures.shape[0]
        #x = self.nodesEmbedding(x[nDrug:])
        #x = x.squeeze(1)

        # xDrug = self.L1(drugFeatures)
        # xDrug = self.actL1(xDrug)
        # xDrug = self.L2(xDrug)
        # xDrug = self.actL2(xDrug)
        #
        # x = torch.cat((xDrug, x), dim=0)
        #
        #
        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # x = F.relu(x)

        x = self.nodesEmbedding(x)
        drugEmbedding = x[drugNodes]
        seEmbedding = x[seNodes]


        return drugEmbedding, seEmbedding, x


    def cal(self, drugE, seE):
        return torch.matmul(drugE, seE.t())
