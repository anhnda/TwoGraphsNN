import config
from utils import utils, drawMol
import numpy as np
import torch



def getCumSumAnchorFromBatchData(batch):
    if batch.__slices__ is None:
        raise RuntimeError(
            ('Cannot reconstruct data list from batch because the batch '
             'object was not created using Batch.from_data_list()'))

    keys = [key for key in batch.keys if key[-5:] != 'batch']
    cumsum = {key: 0 for key in keys}
    nodesumDict = dict()
    nodesumDict[0] = 0
    for i in range(len(batch.__slices__[keys[0]]) - 1):
        data = batch.__data_class__()
        for key in keys:
            data[key] = batch[key].narrow(
                data.__cat_dim__(key, batch[key]), batch.__slices__[key][i],
                batch.__slices__[key][i + 1] - batch.__slices__[key][i])
            data[key] = data[key] - cumsum[key]
            cumsum[key] += data.__inc__(key, data[key])
            if key == 'edge_index':
                nodesumDict[i + 1] = cumsum[key]

    return nodesumDict


def getRef(ext_batch_edges, batch):
    num_batch_graph = int(torch.max(batch)) + 1
    assert num_batch_graph == len(ext_batch_edges)
    batchGraph2Nodes = [[] for i in range(num_batch_graph)]
    for nodeId, batchGraph in enumerate(batch):
        batchGraph2Nodes[int(batchGraph)].append(nodeId)

    dDrugEdgeIndexAnchor = dict()
    dDrugNumNode = dict()
    dDrug2NodeIds = dict()

    dDrugEdgeIndexAnchor[0] = 0
    for i in range(num_batch_graph):
        targetNodeIds = batchGraph2Nodes[i]
        srcNodeIndex = ext_batch_edges[i]
        # srcNodeIds = [j + nInner for j in srcNodeIndex]
        dDrugNumNode[i] = len(targetNodeIds)
        dDrugEdgeIndexAnchor[i + 1] = len(targetNodeIds) * len(srcNodeIndex) + dDrugEdgeIndexAnchor[i]
        dDrug2NodeIds[i] = targetNodeIds

    return dDrug2NodeIds, dDrugNumNode, dDrugEdgeIndexAnchor


def run():
    # Load data
    bioLoader = utils.load_obj("%s/bioLoader" % config.SAVEMODEL_DIR)
    # all_x = utils.load_obj("%s/all_x" % config.SAVEMODEL_DIR)
    # all_edges = utils.load_obj("%s/all_edges" % config.SAVEMODEL_DIR)
    # anchor2 = utils.load_obj("%s/anchor2" % config.SAVEMODEL_DIR)
    ext_edges = utils.load_obj("%s/ext_edges" % config.SAVEMODEL_DIR)
    batch = utils.load_obj("%s/batch" % config.SAVEMODEL_DIR)
    # oalphal = utils.load_obj("%s/oalpha" % config.SAVEMODEL_DIR)
    ealphal = utils.load_obj("%s/ealpha" % config.SAVEMODEL_DIR)
    ealphal = ealphal.data.numpy()
    np.savetxt("%s/arr_ealpha.txt" % config.SAVEMODEL_DIR, ealphal)
    nodeSumDict = getCumSumAnchorFromBatchData(bioLoader.graphBatch)

    print(len(nodeSumDict))
    print(nodeSumDict)

    dDrug2NodeIds, dDrugNumNode, dDrugEdgeIndexAnchor = getRef(ext_edges, batch)

    drugId = 15

    drugProteinList = ext_edges[drugId]

    proteinId = drugProteinList[0]
    print("DrugId: ", drugId)
    print("Select one of proteins")
    print(drugProteinList)
    print("Current: ", proteinId)
    # Find protein Position

    smile = bioLoader.drugId2SMILE[drugId]
    drawMol.plotX("DX", smile, withIndex=True)

    pos = -1
    for i, v in enumerate(drugProteinList):
        if v == proteinId:
            pos = i
            break
    if pos == -1:
        print("Wrong protein", proteinId, " for drug: ", drugId)


        exit(-1)

    # For external link indices
    startEdgeIndex = dDrugEdgeIndexAnchor[drugId] + pos * len(dDrug2NodeIds[drugId])
    extDrugProteinEdgeIndices = []
    for i in range(len(dDrug2NodeIds[drugId])):
        extDrugProteinEdgeIndices.append(startEdgeIndex + i)
    extWeights = ealphal[extDrugProteinEdgeIndices]
    extWeights = np.squeeze(extWeights)
    nodeOriginalId = [v - nodeSumDict[drugId] for v in dDrug2NodeIds[drugId]]
    for i in range(len(nodeOriginalId)):
        print(extWeights[i], nodeOriginalId[i])



if __name__ == "__main__":
    run()
