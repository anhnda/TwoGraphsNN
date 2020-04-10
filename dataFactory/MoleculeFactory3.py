from pysmiles import read_smiles
from torch_geometric.data import Data, Batch

from torch_geometric.utils import to_undirected
from pysmiles import read_smiles
from utils import utils
import torch
import numpy as np
import config


class ModeculeFactory3:
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

    def createBatchGraph(self, atomOffset, proteinOffset, nProtein, drug2ProteinList):
        self.N_ATOM = self.getNumAtom()
        self.N_FEATURE = self.N_ATOM
        cc = 0

        proteinGraph2EdgeIndex = dict()
        proteinGraph2NodeVecIndex = dict()

        drugProteinGraphList = list()

        for proteinId in range(nProtein):
            proteinGraphEdge = utils.get_insert_key_dict(proteinGraph2EdgeIndex, proteinId, [])
            proteinGraphEdge.append([0, 0])
            proteinGraphNodeVec = utils.get_insert_key_dict(proteinGraph2NodeVecIndex, proteinId, [])
            proteinGraphNodeVec.append(proteinId +proteinOffset)

        for drugId, modeculeInfo in enumerate(self.moleculeList):
            nodeFeatures, edgIndex, edgeAttr = modeculeInfo
            nodeVecs = []
            atomIdList = []
            for nodeFeature in nodeFeatures:
                element, atomId, charger, aromatic, hcount = nodeFeature
                nodeVecs.append(atomId + atomOffset)
                atomIdList.append(atomId)

            cc += len(nodeFeatures)
            newEdgIndex = []
            for edge in edgIndex:
                i1, i2 = edge
                newEdgIndex.append([i1, i2])

            for proteinId in drug2ProteinList[drugId]:
                proteinId = int(proteinId)
                nodeVecs.append(proteinId + proteinOffset)
                for nodeId in range(len(nodeFeatures)):
                    v = np.random.uniform()
                    if v < config.CROSS_PROB and config.CROSS_PROB > 0:
                        newEdgIndex.append([proteinId + self.N_ATOM, nodeId])
                        edgeAttr.append([4])
                        proteinGraphEdge = utils.get_dict(proteinGraph2EdgeIndex, proteinId, -1)
                        proteinGraphEdge.append([atomIdList[nodeId], 0])

                        proteinGraphNodeVec = utils.get_dict(proteinGraph2NodeVecIndex, proteinId, -1)
                        proteinGraphNodeVec.append(atomIdList[nodeId] + atomOffset)

            nodeVecs = np.asarray(nodeVecs)
            nodeVecs = torch.from_numpy(nodeVecs).long()
            newEdgIndex = torch.from_numpy(np.asarray(newEdgIndex)).long().t().contiguous()
            # edgeAttr = torch.from_numpy(np.asarray(edgeAttr)).float()

            data = Data(x=nodeVecs, edge_index=newEdgIndex)
            drugProteinGraphList.append(data)


        for proteinGraphId in range(nProtein):

            proteinGraphEdge = proteinGraph2EdgeIndex[proteinGraphId]
            proteinGraphEdge = torch.from_numpy(np.asarray(proteinGraphEdge)).long().t().contiguous()

            nodeVecProteinGraph = proteinGraph2NodeVecIndex[proteinGraphId]
            nodeVecProteinGraph = torch.from_numpy(np.asarray(nodeVecProteinGraph)).long()
            data = Data(x=nodeVecProteinGraph, edge_index=proteinGraphEdge)
            drugProteinGraphList.append(data)

        self.graphList = drugProteinGraphList

        batch = Batch.from_data_list(drugProteinGraphList)
        print("Batch molecular graph completed.")
        print("Total: ", cc, len(self.moleculeList), cc * 1.0 / len(self.moleculeList))

        return batch
