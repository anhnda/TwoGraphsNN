from pysmiles import read_smiles
from torch_geometric.data import Data, Batch

from torch_geometric.utils import to_undirected
from pysmiles import read_smiles
from utils import utils
import torch
import numpy as np
import config

class ModeculeFactory:
    def __init__(self):
        self.__atomElement2Id = dict()
        self.moleculeList = list()
        self.smile2Graph = utils.load_obj(config.SMILE2GRAPH)

    def getAtomIdFromElement(self, ele):
        return utils.get_update_dict_index(self.__atomElement2Id, ele)

    def convertSMILE2Graph(self, smile):
        mol = self.smile2Graph[smile]
        nodes = mol._node
        edges = mol._adj
        nodeFeatures = []
        if len(nodes) == 0:
            print("Wrong")
            print(smile)
            exit(-1)

        keys = nodes.keys()
        keys = sorted(keys)

        mapKeys = dict()
        for k in keys:
            mapKeys[k] = len(mapKeys)

        for nodeId in keys:
            nodeDict = nodes[nodeId]
            element = nodeDict['element']
            atomId = self.getAtomIdFromElement(element)

            charger = nodeDict['charge']
            aromatic = nodeDict['aromatic']
            hcount = nodeDict['hcount']
            nodeFeature = [element, atomId, charger, aromatic, hcount]
            nodeFeatures.append(nodeFeature)

        edgeIndex = []
        edgeAttr = []

        for nodeId, nextNodes in edges.items():
            for nextNodeId, edgeInfo in nextNodes.items():
                edgeIndex.append([mapKeys[nodeId], mapKeys[nextNodeId]])
                edgeAttr.append([edgeInfo['order']])

        return [nodeFeatures, edgeIndex, edgeAttr]

    def addSMILE(self, smile):
        self.moleculeList.append(self.convertSMILE2Graph(smile))

    def createBatchGraph(self):
        self.N_ATOM = len(self.__atomElement2Id)
        self.N_FEATURE = self.N_ATOM + 3
        graphList = list()
        for modeculeInfo in self.moleculeList:
            nodeFeatures, edgIndex, edgeAttr = modeculeInfo
            nodeVecs = []
            for nodeFeature in nodeFeatures:

                element, atomId, charger, aromatic, hcount = nodeFeature

                nodeVec = np.zeros(self.N_FEATURE)

                nodeVec[atomId] = 1
                nodeVec[self.N_ATOM + 0] = charger
                nodeVec[self.N_ATOM + 1] = int(aromatic)
                nodeVec[self.N_ATOM + 2] = hcount
                nodeVecs.append(nodeVec)

            nodeVecs = np.vstack(nodeVecs)
            nodeVecs = torch.from_numpy(nodeVecs).float()
            edgIndex = torch.from_numpy(np.asarray(edgIndex)).long().t().contiguous()
            edgeAttr = torch.from_numpy(np.asarray(edgeAttr)).float()

            data = Data(x=nodeVecs, edge_index=edgIndex, edge_attr=edgeAttr)
            graphList.append(data)

        self.graphList = graphList

        batch = Batch.from_data_list(graphList)

        return batch
