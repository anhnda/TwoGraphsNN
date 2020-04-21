from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils.num_nodes import maybe_num_nodes

import torch_geometric
import torch
import numpy as np

import itertools
import random


def create_ext_edges(x, edge_index, xref):
    if xref is None:
        return x, edge_index, None, None

    nNodeOrig, dim1 = x.shape
    nNodeRefer, dim2 = xref.shape
    assert nNodeOrig >= nNodeRefer
    assert dim1 == dim2

    x = torch.cat([x, xref], dim=0)
    srcNodeIds = np.arange(nNodeOrig, nNodeOrig + nNodeRefer)
    targetNodeIds = np.arange(0, nNodeRefer)
    edge2 = np.vstack([srcNodeIds, targetNodeIds])
    extEdges = torch.from_numpy(edge2).long()
    allEdges = torch.cat((edge_index, extEdges), dim=1)

    oldAnchor = edge_index.shape[1]

    return x, allEdges, oldAnchor, nNodeRefer
