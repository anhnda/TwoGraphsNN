import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv
from torch import sigmoid
import config


class Net(torch.nn.Module):
    def __init__(self, numNode=10000):
        super(Net, self).__init__()

        self.conv1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.conv2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.conv3 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)  # SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.nodesEmbedding = torch.nn.Embedding(num_embeddings=numNode + 1, embedding_dim=config.EMBED_DIM)
        self.nodesEmbedding.weight.data.uniform_(0.001, 0.3)




    def forward2(self, data, drugNodes, seNodes):
        x, edge_index = data.x, data.edge_index
        x = self.nodesEmbedding(x)
        x = x.squeeze(1)
        # x = self.conv1(x, edge_index)

        # x = self.conv1(x, edge_index)
        # x = F.relu(x)
        # x = self.conv2(x, edge_index)
        # # x = sigmoid(x)
        # x = F.relu(x)

        # x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))

        # drugEmbedding = x[drugNodes]
        # seEmbedding = x[seNodes]

        drugEmbedding = self.nodesEmbedding(drugNodes)
        seEmbedding = self.nodesEmbedding(seNodes)
        # re = torch.sigmoid(re)
        return drugEmbedding, seEmbedding, x

        # x = F.dropout(x, p=0.5, training=self.training)


    def forward(self, data, drugNodes, seNodes):
        x, edge_index = data.x, data.edge_index
        x = self.nodesEmbedding(x)
        x = x.squeeze(1)

        x = self.conv1(x, edge_index)
        # x = sigmoid(x)
        # x = F.relu(x)


        x = self.conv2(x, edge_index)
        # x = sigmoid(x)

        # x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        # x = F.relu(self.conv3(x, edge_index))

        drugEmbedding = x[drugNodes]
        seEmbedding = x[seNodes]

        # drugEmbedding = self.nodesEmbedding(drugNodes)
        # seEmbedding = self.nodesEmbedding(seNodes)
        # re = torch.sigmoid(re)
        return drugEmbedding, seEmbedding, x

