import torch
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.modules import EdgeConv
from torch import sigmoid
import config

from torch.nn import Linear


from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



class Net4_2(torch.nn.Module):
    def __init__(self, numNode=10000, numAtomFeature=0):
        super(Net4_2, self).__init__()

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



        # Molecule graph neural net

        self.mlinear1 = Linear(numAtomFeature, config.EMBED_DIM * 2)
        self.mact1 = F.relu
        self.mlinear2 = Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.mact2 = F.relu

        self.conv1 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.pool1 = TopKPooling(config.EMBED_DIM, ratio=0.8)
        self.conv2 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.pool2 = TopKPooling(config.EMBED_DIM, ratio=0.8)
        self.conv3 = SAGEConv(config.EMBED_DIM, config.EMBED_DIM)
        self.pool3 = TopKPooling(config.EMBED_DIM, ratio=0.8)
        self.lin1 = torch.nn.Linear(config.EMBED_DIM * 2, config.EMBED_DIM)
        self.lin2 = torch.nn.Linear(config.EMBED_DIM, config.EMBED_DIM)


        self.bn1 = torch.nn.BatchNorm1d(128)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()




    def forward(self, x, drugEdges, seEdges, drugNodes, seNodes, drugGraphBatch, nDrug):

        x = self.nodesEmbedding(x[nDrug:])
        x = x.squeeze(1)


        xDrug, edge_index, batch = drugGraphBatch.x, drugGraphBatch.edge_index, drugGraphBatch.batch

        xDrug = self.mact1(self.mlinear1(xDrug))
        xDrug = self.mact2(self.mlinear2(xDrug))

        xDrug = F.relu(self.conv1(xDrug, edge_index))

        v  = self.pool1(xDrug, edge_index, None, batch)
        xDrug, edge_index, _, batch, _, _ = v
        x1 = torch.cat([gmp(xDrug, batch), gap(xDrug, batch)], dim=1)

        xDrug = F.relu(self.conv2(xDrug, edge_index))

        xDrug, edge_index, _, batch, _, _ = self.pool2(xDrug, edge_index, None, batch)
        x2 = torch.cat([gmp(xDrug, batch), gap(xDrug, batch)], dim=1)

        xDrug = F.relu(self.conv3(xDrug, edge_index))

        xDrug, edge_index, _, batch, _, _ = self.pool3(xDrug, edge_index, None, batch)
        x3 = torch.cat([gmp(xDrug, batch), gap(xDrug, batch)], dim=1)

        xDrug = x1 + x2 + x3

        xDrug = self.lin1(xDrug)
        xDrug = self.act1(xDrug)
        xDrug = self.lin2(xDrug)
        xDrug = self.act2(xDrug)



        x = torch.cat((xDrug, x), dim=0)

        # Conv Drug:
        # x = self.convD1(x, drugEdges)
        # x = F.relu(x)
        # x = self.convD2(x, drugEdges)
        # x = F.relu(x)

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
