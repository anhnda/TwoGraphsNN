from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils.num_nodes import maybe_num_nodes

import torch_geometric
import torch
import numpy as np


def create_ext_edges(x, edge_index, ext_tensor=None, extProb=0.001, ext_weight=None, size=None):
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
