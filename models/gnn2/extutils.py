from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils.num_nodes import maybe_num_nodes

import torch_geometric
import torch
import numpy as np

import itertools
import random

def create_ext_edges(x, edge_index, ext_embed, ext_batch_edges, batch, extProb):
    if ext_embed is not None:
        nOuter, dimOuter = ext_embed.shape
        nInner, dimInner = x.shape
        assert dimOuter == dimInner
        x = torch.cat([x, ext_embed], dim=0)
        extEdges = []

        # Garther node by batch

        num_batch_graph= int(torch.max(batch)) + 1
        assert num_batch_graph == len(ext_batch_edges)
        batchGraph2Nodes = [[] for i in range(num_batch_graph)]
        for nodeId, batchGraph in enumerate(batch):
            batchGraph2Nodes[int(batchGraph)].append(nodeId)

        if extProb > 0:
            for i in range(num_batch_graph):
                targetNodeIds = batchGraph2Nodes[i]
                srcNodeIndex = ext_batch_edges[i]
                srcNodeIds = [j + nInner for j in srcNodeIndex]
                edges = list(itertools.product(srcNodeIds, targetNodeIds))
                if extProb < 1:
                    edges = random.sample(edges, int(len(edges) * extProb))
                edges = np.asarray(edges).transpose()
                if len(edges.shape) == 1:
                    # Missing edges
                    continue
                extEdges.append(edges)
            extEdges = np.concatenate(extEdges, axis=1)
            extEdges = torch.from_numpy(extEdges).long()
            allEdges = torch.cat((edge_index, extEdges), dim=1)
        else:
            allEdges = edge_index
        oldAnchor = edge_index.shape[1]

        return x, allEdges, oldAnchor, nOuter



def create_ext_edges2(x, edge_index, ext_tensor=None, extProb=0.001, ext_weight=None, size=None):
    if ext_tensor is not None:
        nOuter, dimOuter = ext_tensor.shape
        nInner, dimInner = x.shape
        assert dimOuter == dimInner
        x = torch.cat([x, ext_tensor], dim=0)
        extEdges = []
        targetArray = np.arange(0, nInner, 1, dtype=int)
        if ext_weight is None:
            ext_weight = np.ones(nOuter)
        if extProb > 0:
            for i in range(nOuter):
                srcNodeId = i + nInner
                srcArray = np.ndarray(nInner, dtype=int)
                srcArray.fill(srcNodeId)
                selectedEdge = np.random.choice(nInner, max(2, int(extProb * nInner * ext_weight[i])))
                edgeArray = np.vstack([srcArray, targetArray])
                edgeArray = edgeArray[:, selectedEdge]
                extEdges.append(edgeArray)
            extEdges = np.concatenate(extEdges, axis=1)
            extEdges = torch.from_numpy(extEdges).long()
            allEdges = torch.cat((edge_index, extEdges), dim=1)
        else:
            allEdges = edge_index
        oldAnchor = edge_index.shape[1]

        return x, allEdges, oldAnchor, nOuter
