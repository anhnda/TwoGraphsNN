from torch_geometric.data import Data, Batch
import torch
import numpy as np


class Batch2:
    def __init__(self, x=None, edge_index=None, batch=None):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch


class NestedGraph:
    def __init__(self):
        self.level2Graphs = dict()
        self.numLevel = 0

    def initLevel0(self, size0):
        self.numLevel = 0
        graphLevel0 = [None for i in range(size0)]
        self.level2Graphs[self.numLevel] = graphLevel0

    def addNewLevel(self):
        self.numLevel += 1
        self.level2Graphs[self.numLevel] = []

    def getGraphListAtLevel(self, level):
        return self.level2Graphs[level]

    def getGraphListAtCurrentLevel(self):
        return self.getGraphListAtLevel(self.numLevel)

    def addGraph(self, graph):
        self.getGraphListAtCurrentLevel().append(graph)

    def getSizeOfLevel(self, level):
        return len(self.getGraphListAtLevel(level))

    def createBatchForLevel(self, level):
        graphList = self.getGraphListAtLevel(level)
        allEdgeIndex = []
        totalNode = self.getSizeOfLevel(level - 1)
        batch = np.ndarray(totalNode, dtype=int)
        batch.fill(-1)
        for i, graph in enumerate(graphList):
            allEdgeIndex.append(graph.edge_index)
            for j in graph.edge_index[0]:
                batch[j] = i
            for j in graph.edge_index[1]:
                batch[j] = i
        allEdgeIndex = torch.cat(allEdgeIndex, dim=1)
        batch = torch.from_numpy(batch).long()
        batchData = Batch2(None, allEdgeIndex, batch)
        return batchData, totalNode


def convertFromBatch(batch):
    if batch.__slices__ is None:
        raise RuntimeError(
            ('Cannot reconstruct data list from batch because the batch '
             'object was not created using Batch.from_data_list()'))
    keys = [key for key in batch.keys if key[-5:] != 'batch']
    # cumsum = {key: 0 for key in keys}
    nodesumDict = dict()
    nodesumDict[0] = 0
    graphs = []
    for i in range(len(batch.__slices__[keys[0]]) - 1):
        data = batch.__data_class__()
        key = 'edge_index'
        edge_index = batch[key].narrow(
            data.__cat_dim__(key, batch[key]), batch.__slices__[key][i],
            batch.__slices__[key][i + 1] - batch.__slices__[key][i])

        graph = Data(edge_index=edge_index)
        graphs.append(graph)

    return graphs


def convertBioLoaderToNestedGraph(bioLoader):
    nestedGraph = NestedGraph()
    # Level 0: Atom in Batch + Proteins + Pathways
    # Molecular graph:
    atom_x, atom_edge_index, atom_batch = bioLoader.graphBatch.x, bioLoader.graphBatch.edge_index, bioLoader.graphBatch.batch

    # Total node at level 0:
    size0 = atom_batch.shape[0] + bioLoader.nProtein + bioLoader.nPathway
    nestedGraph.initLevel0(size0)

    # Level one:
    nestedGraph.addNewLevel()

    # Molecular graph
    nMolecularGraph = torch.max(atom_batch) + 1

    molecularGraphs = convertFromBatch(bioLoader.graphBatch)
    # print (len(molecularGraphs),  nMolecularGraph)
    # print (torch.min(atom_batch), torch.max(atom_batch))
    assert len(molecularGraphs) == nMolecularGraph
    for graph in molecularGraphs:
        nestedGraph.addGraph(graph)

    # For proteins and pathway empty grah
    OFFSET = atom_batch.shape[0]
    for i in range(OFFSET, OFFSET + bioLoader.nProtein + bioLoader.nPathway):
        edge_index = torch.from_numpy(np.asarray([[i], [i]])).long()
        graph = Data(edge_index=edge_index)
        nestedGraph.addGraph(graph)

    # Level two:
    nestedGraph.addNewLevel()
    MOL_GRAPH_OFFSET = 0
    PROTEIN_GRAPH_OFFSET = nMolecularGraph
    PATHWAY_GRAPH_OFFSET = PROTEIN_GRAPH_OFFSET + bioLoader.nProtein
    edge_index = []
    # Add drug - protein
    for drugId in range(bioLoader.nDrug):
        proteinIds = bioLoader.drugId2ProteinIndices[drugId]
        for proteinId in proteinIds:
            edge_index.append([drugId + MOL_GRAPH_OFFSET, proteinId + PROTEIN_GRAPH_OFFSET])
            edge_index.append([proteinId + PROTEIN_GRAPH_OFFSET, drugId + MOL_GRAPH_OFFSET])

    # Add protein - pathways:
    matProtein2Pathway = bioLoader.matProtein2Pathway
    for proteinId in range(bioLoader.nProtein):
        pathways = np.nonzero(matProtein2Pathway[proteinId])[0]
        for pathwayId in pathways:
            edge_index.append([proteinId + PROTEIN_GRAPH_OFFSET, pathwayId + PATHWAY_GRAPH_OFFSET])
            edge_index.append([pathwayId + PATHWAY_GRAPH_OFFSET, proteinId + PROTEIN_GRAPH_OFFSET])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    x = [[i] for i in range(nMolecularGraph + bioLoader.nProtein + bioLoader.nPathway)]
    x = torch.from_numpy(np.asarray(x)).long()

    graph = Data(x=x, edge_index=edge_index)

    nestedGraph.addGraph(graph)
    print("Converting Bioloader to NestedGraph completed")
    return nestedGraph


if __name__ == "__main__":
    from dataFactory.loader import BioLoader5P2

    bioLoader = BioLoader5P2()
    trainPath = BioLoader5P2.getPathNTIMESIFold(0, 0)
    bioLoader.createTrainTestVal(trainPath)
    nestedGraph = convertBioLoaderToNestedGraph(bioLoader)
    batch1 = nestedGraph.createBatchForLevel(1)

    print(nestedGraph)
