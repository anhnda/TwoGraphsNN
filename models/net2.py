import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear


class Net2(torch.nn.Module):
    def __init__(self, numNode=10000):
        super(Net2, self).__init__()

        self.convD1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convD2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)


        self.convS1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.convS2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)



        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)

    def forward(self, x, drugEdges, seEdges, drugNodes, seNodes):
        x = self.nodesEmbedding(x)
        x = x.squeeze(1)
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
