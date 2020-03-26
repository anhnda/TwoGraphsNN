from pysmiles import read_smiles
from torch_geometric.data import Data, Batch

from torch_geometric.utils import to_undirected
from pysmiles import read_smiles
from utils import utils
import torch
import numpy as np
import config


class ModeculeFactory2:
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

    def getNumAtom(self):
        return len(self.__atomElement2Id)

    def createBatchGraph(self, atomOffset, proteinOffset, nProtein):
        self.N_ATOM = self.getNumAtom()
        self.N_FEATURE = self.N_ATOM
        graphList = list()
        for modeculeInfo in self.moleculeList:
            nodeFeatures, edgIndex, edgeAttr = modeculeInfo
            nodeVecs = []
            for nodeFeature in nodeFeatures:
                element, atomId, charger, aromatic, hcount = nodeFeature
                nodeVecs.append(atomId + atomOffset)


            newEdgIndex = []
            for edge in edgIndex:
                i1, i2 = edge
                newEdgIndex.append([i1, i2])

            # for proteinId in range(nProtein):
            #     nodeVecs.append(proteinId+proteinOffset)
            #     for nodeId in range(len(nodeFeatures)):
            #         newEdgIndex.append([proteinId, nodeId])
            #         edgeAttr.append([4])

            nodeVecs = np.asarray(nodeVecs)
            nodeVecs = torch.from_numpy(nodeVecs).long()
            newEdgIndex = torch.from_numpy(np.asarray(newEdgIndex)).long().t().contiguous()
            edgeAttr = torch.from_numpy(np.asarray(edgeAttr)).float()

            data = Data(x=nodeVecs, edge_index=newEdgIndex, edge_attr=edgeAttr)
            graphList.append(data)

        self.graphList = graphList

        batch = Batch.from_data_list(graphList)
        print("Batch molecular graph completed.")

        return batch
