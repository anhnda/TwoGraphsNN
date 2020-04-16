from dataFactory import nestedGraph

def createLevel2Data(bioLoader):
    nestedGraphData = nestedGraph.convertBioLoaderToNestedGraph(bioLoader)
    level2Graph = nestedGraphData.getGraphListAtLevel(2)[0]
    x = level2Graph.x


