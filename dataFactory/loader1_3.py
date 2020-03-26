import config
from utils import utils
import numpy as np
from dataFactory import loadingMap

import torch
from torch_geometric.data import Data

from torch_geometric.utils import to_undirected


class BioLoader1:
    def __init__(self):
        self.drugInchiKey2Id = dict()
        self.sideEffectName2Id = dict()
        self.drugId2Features = dict()
        self.drugId2Inchikey = dict()
        self.drugId2PathwayProfiles = dict()

        self.pathway2Id = dict()
        self.protein2Id = dict()
        self.id2ProteinName = dict()
        self.pathwayId2Des = dict()
        self.drugName2Inchi = None
        self.drugInchi2Name = None
        self.drugName2SMILE = None
        self.drugId2Index = None
        self.drugIndex2Id = None
    def loadValidInchi(self):
        path = BioLoader1.getPathIFold(0)
        fin = open(path)
        self.validInchi = set()
        while True:
            line = fin.readline()
            if line == "":
                break
            self.validInchi.add(line.split("|")[0])
    def loadPathWay(self):
        allPathway = config.KEGG_PATHWAY_FILE
        fin = open(allPathway)
        while True:
            line = fin.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("\t")
            pathwayName, des = parts[0], parts[1]
            pathId = utils.get_update_dict_index(self.pathway2Id, pathwayName)
            self.pathwayId2Des[pathId] = des
        fin.close()
        self.nPathway = len(self.pathway2Id)
        self.id2Pathway = utils.reverse_dict(self.pathway2Id)

    def loadDrugChemFeatures(self):

        def loadMorgan():
            fin = open(config.INCHI_MORGAN_FILE)
            d = {}
            while True:
                line = fin.readline()
                if line == "":
                    break
                parts = line.strip().split("|")
                inchi = parts[0]
                morganString = parts[1].split(",")
                fingerPrint = np.zeros(config.CHEM_FINGERPRINT_SIZE)
                for i, v in enumerate(morganString):
                    if float(v) == 1:
                        fingerPrint[i] = 1
                d[inchi] = fingerPrint
            fin.close()
            return d

        if config.MORGAN:
            dInchi2Features = loadMorgan()
        else:
            dInchi2Features = utils.load_obj(config.PUBCHEM_FILE)
        inchiKeyList = sorted(list(dInchi2Features.keys()))
        dBit2Inchi = dict()
        self.loadValidInchi()
        for k in inchiKeyList:
            if k not in self.validInchi:
                continue
            v = dInchi2Features[k]
            chemId = utils.get_update_dict_index(self.drugInchiKey2Id, k)
            self.drugId2Features[chemId] = v
            nonZeros = np.nonzero(v)[0]
            # if k == "QTGIAADRBBLJGA-UHFFFAOYSA-N":
            #    print nonZeros
            for bit in nonZeros:
                vInchis = utils.get_insert_key_dict(dBit2Inchi, bit, [])
                vInchis.append(k)

    def loadDrug2Proteins(self):
        self.drug2ProteinList = loadingMap.loadDrugProteinMap()
        proteinListList = self.drug2ProteinList.values()
        protensSets = set()
        for proteins in proteinListList:
            for protein in proteins:
                if protein != "":
                    protensSets.add(protein)

        proteinList = list(protensSets)
        proteinList = sorted(proteinList)
        for protein in proteinList:
            utils.get_update_dict_index(self.protein2Id, protein)

        self.id2ProteinName = utils.reverse_dict(self.protein2Id)
        # fUniProtIdFile = open("%s/UniProtID.dat" % config.OUTPUT_DIR, "w")
        # for v in self.protein2Id.keys():
        #     fUniProtIdFile.write("%s\n" % v)
        # fUniProtIdFile.close()

        self.nProtein = len(self.protein2Id)
        self.drugId2ProteinIndices = dict()
        for k, v in self.drug2ProteinList.items():
            drugId = utils.get_dict_index_only(self.drugInchiKey2Id, k)
            if drugId != -1:
                proteinIdList = []
                for vv in v:
                    proteinIdList.append(self.protein2Id[vv])


                self.drugId2ProteinIndices[drugId] = proteinIdList

    def loadDART(self):
        def loadMapFromFile(path):
            d = dict()
            f = open(path)
            while True:
                line = f.readline()
                if line == "":
                    break
                parts = line.strip().lower().split("|")
                d[parts[0]] = parts[1]
            f.close()
            return d

        dDrugDart2DrugBank = loadMapFromFile(config.DART_DRUG_MATCH_FILE)
        dSeDart2SIDER = loadMapFromFile(config.DART_SE_MATCH_FILE)
        print(len(dDrugDart2DrugBank), len(dSeDart2SIDER))

        dDartBenchMark = dict()
        f = open(config.DART_RAW_FILE)
        proteinSetX = set()
        drugSetX = set()
        seSetX = set()
        while True:
            line = f.readline()
            if line == "":
                break
            # print(line)
            parts = line.strip().split("#")
            proteinUniProtId = parts[0]
            ses = parts[1].split(",")
            drugs = parts[2].split(",")

            seIds = []
            drugIds = []
            for se in ses:
                seSider = utils.get_dict(dSeDart2SIDER, se, "")
                if seSider != "":
                    seId = utils.get_dict(self.sideEffectName2Id, seSider, -1)
                    if seId != -1:
                        seSetX.add(seId)
                        seIds.append(seId)

            for drug in drugs:
                drugBankName = utils.get_dict(dDrugDart2DrugBank, drug, "")
                if drugBankName != "":
                    drugInchi = utils.get_dict(self.drugName2Inchi, drugBankName, "None")
                    drugId = utils.get_dict(self.drugInchiKey2Id, drugInchi, -1)
                    if drugId != -1:
                        drugSetX.add(drugId)
                        drugIds.append(drugId)
            proteinId = utils.get_dict(self.protein2Id, proteinUniProtId, -1)

            if proteinId != -1 and len(seIds) > 0 and len(drugIds) > 0:
                proteinSetX.add(proteinId)
                for seId in seIds:
                    for drugId in drugIds:
                        pair = (drugId, seId)
                        proteinsofPair = utils.get_insert_key_dict(dDartBenchMark, pair, set())
                        proteinsofPair.add(proteinId)
        self.dartBenchMark = dDartBenchMark

        # Write out:
        fout = open("%s/DART_BENCHMARK.dat" % config.OUTPUT_DIR, "w")
        for k, v in dDartBenchMark.items():
            drugId, seId = k
            drugName = self.drugInchi2Name[self.drugId2Inchikey[drugId]]
            seName = self.id2SideEffectNames[seId]
            proList = []
            for p in v:
                proList.append(self.id2ProteinName[p])
            fout.write("%s|%s|%s\n" % (drugName, seName, ",".join(proList)))
        fout.close()
        print("DART Info: Drug, Se, Protein, Pair: ", len(drugSetX), len(seSetX), len(proteinSetX), len(dDartBenchMark))
        return dDartBenchMark

    def loadSideEffectOntology(self):

        import re
        fin = open(config.SE_NAME_PROP_FILE)
        dSeName2Prop = dict()
        selectedSeNames = set()
        while True:
            line = fin.readline()
            if line == "":
                break
            line = line.strip()
            line = re.sub("\s\s+", " ", line)
            parts = line.strip().split("\t")
            seName = parts[0]
            prop = float(parts[1])
            dSeName2Prop[seName] = prop
            if prop >= config.SE_PROP_THRES:
                selectedSeNames.add(seName)

        print("Num selected se: ", len(selectedSeNames), "Thres: ", config.SE_PROP_THRES)

        dMeddra2SeNames = dict()
        fin = open(config.SE_MEDDRA_NAME_FILE)
        while True:
            line = fin.readline()
            if line == "":
                break
            line = line.strip()
            parts = line.split("\t")
            dMeddra2SeNames[parts[0]] = parts[1]
        fin.close()

        selectedSeMeddraSet = set()
        self.seName2MeddraAll = utils.reverse_dict(dMeddra2SeNames)
        self.seName2MeddraSelected = dict()
        for seName in selectedSeNames:
            selectedSeMeddraSet.add(self.seName2MeddraAll[seName])
            self.seName2MeddraSelected[seName] = self.seName2MeddraAll[seName]

        self.selectedSeNames = selectedSeNames

        meddraSet = set()
        parentDict = dict()
        seMeddraIds = set()
        # Load leaves:
        f = open(config.SE_ONTOLOTY_FILE_1)
        while True:
            line = f.readline()
            if line == "":
                break

            line = line.strip()
            parts = line.split("|")
            childMeddraId = parts[0]
            parrentMeddraIds = parts[1].split(",")
            if childMeddraId not in selectedSeMeddraSet:
                continue
            seMeddraIds.add(childMeddraId)
            meddraSet.add(childMeddraId)
            currentParentSet = utils.get_insert_key_dict(parentDict, childMeddraId, set())
            for parentId in parrentMeddraIds:
                meddraSet.add(parentId)
                currentParentSet.add(parentId)

        f.close()

        lv1MeddraSet = meddraSet.copy()
        # Load next level:
        for path in [config.SE_ONTOLOGY_PARENT_FILE_1, config.SE_ONTOLOGY_PARENT_FILE_2]:
            f = open(path)
            f.readline()
            while True:
                line = f.readline()
                if line == "":
                    break
                line = line.strip()
                parts = line.split("|")
                childMeddraId = parts[0]
                if childMeddraId not in lv1MeddraSet:
                    continue
                parrentMeddraIds = parts[1].split(",")
                currentParentSet = utils.get_insert_key_dict(parentDict, childMeddraId, set())
                for parentId in parrentMeddraIds:
                    currentParentSet.add(parentId)
                    meddraSet.add(parentId)

            f.close()

        print(len(meddraSet), len(seMeddraIds), len(parentDict))

        self.dMeddra2Id = dict()
        sortedSeIds = sorted(list(seMeddraIds))
        sortedMeddraSet = sorted(list(meddraSet))

        for meddraId in sortedSeIds:
            utils.get_update_dict_index(self.dMeddra2Id, meddraId)
        for meddraId in sortedMeddraSet:
            if meddraId not in seMeddraIds:
                utils.get_update_dict_index(self.dMeddra2Id, meddraId)

        self.dSeName2Id = dict()

        for k, v in self.seName2MeddraSelected.items():
            seId = self.dMeddra2Id[v]
            self.dSeName2Id[k] = seId

        self.meddraSet = meddraSet
        self.seMeddraIds = seMeddraIds
        self.seParentDict = parentDict
        self.dseId2Names = utils.reverse_dict(self.dSeName2Id)

        self.allParents = dict()

        for se in sortedSeIds:
            parent1s = utils.get_dict(parentDict, se, set())
            parents = parent1s.copy()

            for parent1 in parent1s:
                parents.add(parent1)
                p2 = utils.get_dict(parentDict, parent1, set())
                for p in p2:
                    parents.add(p)
                    p3 = utils.get_dict(parentDict, p, set())
                    for pp in p3:
                        parents.add(pp)
            parents.add(se)
            self.allParents[se] = parents

        seMat = np.zeros((len(sortedSeIds), len(sortedMeddraSet)))
        print("X len", len(self.allParents), len(self.dseId2Names))
        for sId in range(len(sortedSeIds)):
            parents = self.allParents[sortedSeIds[sId]]
            for parent in parents:
                seMat[sId, self.dMeddra2Id[parent]] = 1

        self.seMat = seMat

        dSeId2Prop = dict()
        for sId in range(len(sortedSeIds)):
            seName = self.dseId2Names[sId]
            prop = dSeName2Prop[seName]
            dSeId2Prop[sId] = prop

        self.dSeId2Prop = dSeId2Prop

        listRareSe = list()
        STEP_SIZE = 0.01
        for i in range(1, int(config.RARE_THRESHOLD / STEP_SIZE)):
            t = STEP_SIZE * i
            l = list()
            for k, v in dSeId2Prop.items():
                if v <= t:
                    l.append(k)

            l = np.asarray(l)
            listRareSe.append(l)

        self.listRareSe = listRareSe

        seId1 = listRareSe[-1][0]
        seName1 = self.dseId2Names[seId1]
        prop1 = self.dSeId2Prop[seId1]
        print("Id, name, prop: ", seId1, seName1, prop1)

    def createSharedGraph(self):
        NUM_NODES = 0

        # CHEM

        self.loadDrugChemFeatures()
        self.nDrug = len(self.drugInchiKey2Id)
        self.DRUG_OFFSET = NUM_NODES
        NUM_NODES += self.nDrug


        self.loadDrug2Proteins()
        self.PROTEIN_OFFSET = NUM_NODES
        NUM_NODES += self.nProtein

        print(len(self.drugId2ProteinIndices))
        self.loadPathWay()

        self.PATHWAY_OFFSET = NUM_NODES
        NUM_NODES += self.nPathway



        self.loadSideEffectOntology()
        self.nSe = len(self.seMeddraIds)
        self.SE_OFFSET = NUM_NODES
        NUM_NODES += len(self.meddraSet)

        self.NUM_NODES = NUM_NODES
        print("Total nodes: ", self.NUM_NODES)

        ax = np.zeros(self.NUM_NODES, dtype=np.long)
        for i in range(self.NUM_NODES):
            ax[i] = i

        x = torch.from_numpy(ax).long()
        edge_index = []

        # Protein 2 Pathway
        self.mapProtein2Pathway = loadingMap.loadProtein2Pathway()
        self.matProtein2Pathway = np.zeros((self.nProtein, self.nPathway))

        for proteinId in range(self.nProtein):
            proteinName = self.id2ProteinName[proteinId]
            pathways = utils.get_dict(self.mapProtein2Pathway, proteinName, -1)
            if pathways != -1:
                for p in pathways:
                    pathWayId = utils.get_dict(self.pathway2Id, p, -1)
                    if pathWayId != -1:
                        edge_index.append([pathWayId + self.PATHWAY_OFFSET, proteinId + self.PROTEIN_OFFSET])
                        self.matProtein2Pathway[proteinId, pathWayId] = 1






        # Drug 2 Chem Matrix Only:
        self.matDrugChem = np.zeros((self.nDrug, config.CHEM_FINGERPRINT_SIZE))
        for drugId in range(self.nDrug):
            chems = self.drugId2Features[drugId]
            nonZeros = np.nonzero(chems)[0]
            for chemId in nonZeros:
                chemId = int(chemId)
                self.matDrugChem[drugId, chemId] = 1

        # Drug 2 Protein
        self.matDrugProtein = np.zeros((self.nDrug, self.nProtein))

        for drugId in range(self.nDrug):
            proteinIndices = self.drugId2ProteinIndices[drugId]
            for proteinId in proteinIndices:
                proteinId = int(proteinId)
                edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                self.matDrugProtein[drugId, proteinId] = 1

        # Add DART
        if config.DART:
            self.loadDART()
            for k, v in self.dartBenchMark.items():
                drugId, _ = k
                proteinIds = v
                for proteinId in proteinIds:
                    edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                    self.matDrugProtein[drugId, proteinId] = 1


        # SE

        for child, parents in self.seParentDict.items():
            seChildId = utils.get_dict(self.dMeddra2Id, child, -1)
            for parent in parents:
                if parent == "":
                    continue
                seParentId = utils.get_dict(self.dMeddra2Id, parent, -1)
                if seParentId != -1:
                    edge_index.append([seParentId + self.SE_OFFSET, seChildId + self.SE_OFFSET])

        self.x = x
        self.edge_index = edge_index



        # drugPathway:
        self.matDrugPathway = np.zeros((self.nDrug, self.nPathway))
        m = np.dot(self.matDrugProtein, self.matProtein2Pathway)
        m[m >= 1] = 1
        self.matDrugPathway = m

        self.drugFeatures = torch.from_numpy(self.matDrugChem).float()

    def loadTrainTest(self, fPath, allTrain=False):
        fin = open(fPath)
        dDrugId2SideEffect = dict()

        drugTrainIds = list()
        drugTestIds = list()
        currenDrugSet = drugTrainIds

        while True:
            line = fin.readline()
            if line == "":
                break
            line = line.strip()
            if line.startswith("#"):
                if allTrain == False:
                    currenDrugSet = drugTestIds
                continue
            parts = line.split("|")
            inchi = parts[0]
            sideEffects = parts[-1].split(",")
            drugId = utils.get_dict_index_only(self.drugInchiKey2Id, inchi)
            if drugId == -1:
                continue

            currenDrugSet.append(drugId)
            sideEffectsIds = []
            for sd in sideEffects:
                sideEffectId = utils.get_dict_index_only(self.dSeName2Id, sd)
                if sideEffectId != -1:
                    sideEffectsIds.append(sideEffectId)
            dDrugId2SideEffect[drugId] = sideEffectsIds
        fin.close()

        # Add DART
        if config.DART:
            for k, v in self.dartBenchMark.items():
                drugId, seId = k
                seList = dDrugId2SideEffect[drugId]
                if seId not in seList:
                    seList.append(seId)

        self.drugTrainIdList = drugTrainIds
        self.drugTestIdList = drugTestIds
        self.drugId2Ses = dDrugId2SideEffect

        print(len(drugTrainIds), len(drugTestIds), self.nSe)
        trainOutMatrix = np.zeros((len(drugTrainIds), self.nSe), dtype=float)
        rowId = 0
        for drugId in drugTrainIds:
            seIds = dDrugId2SideEffect[drugId]

            for seId in seIds:
                trainOutMatrix[rowId, seId] = 1
            rowId += 1

        testOutMatrix = np.zeros((len(drugTestIds), self.nSe), dtype=float)
        rowId = 0
        for drugId in drugTestIds:
            seIds = dDrugId2SideEffect[drugId]
            for seId in seIds:
                testOutMatrix[rowId, seId] = 1
            rowId += 1

        self.trainOutMatrix = torch.from_numpy(trainOutMatrix).float()
        self.testOutMatrix = torch.from_numpy(testOutMatrix).float()

        inputAll = np.concatenate([self.matDrugChem, self.matDrugProtein, self.matDrugPathway], axis=1)
        self.trainInpMat = inputAll[drugTrainIds, :]
        self.testInpMat = inputAll[drugTestIds, :]

    def createTrainTestGraph(self, fPath):
        self.createSharedGraph()
        self.loadTrainTest(fPath)
        # Add training drug-se edges:
        for drugId in self.drugTrainIdList:
            seIds = self.drugId2Ses[drugId]
            for seId in seIds:
                self.edge_index.append([drugId + self.DRUG_OFFSET, seId + self.SE_OFFSET])
                # self.edge_index.append([seId + self.SE_OFFSET, drugId + self.DRUG_OFFSET])

        edge_index = torch.tensor(self.edge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        if config.UN_DIRECTED:
            edge_index = to_undirected(edge_index)

        self.graphData = Data(x=self.x, edge_index=edge_index)
        drugTrainNodeIds = list()
        drugTestNodeIds = list()
        for drugId in self.drugTrainIdList:
            drugTrainNodeIds.append(drugId + self.DRUG_OFFSET)
        for drugId in self.drugTestIdList:
            drugTestNodeIds.append(drugId + self.DRUG_OFFSET)
        seNodeIds = list()
        for seId in range(self.nSe):
            seNodeIds.append(seId + self.SE_OFFSET)
        self.drugTrainNodeIds = torch.from_numpy(np.asarray(drugTrainNodeIds, dtype=np.long)).long()
        self.drugTestNodeIds = torch.from_numpy(np.asarray(drugTestNodeIds, dtype=np.long)).long()
        self.seNodeIds = torch.from_numpy(np.asarray(seNodeIds, dtype=np.long)).long()
        print("Undirected graph: ", self.graphData.is_undirected(), "Config: ", config.UN_DIRECTED)

    @staticmethod
    def getTrainPathPref():
        pathTrainPref = "%s%s" % (config.PATH_KFOLD_ATC_INCHIKEY_SIDEEFFECT_BYDRUG, config.TRAIN_PREFIX)
        return pathTrainPref

    @staticmethod
    def getPathIFold(iFold):
        pathTrain = "%s%s" % (BioLoader1.getTrainPathPref(), iFold)
        return pathTrain


if __name__ == "__main__":
    bioloader = BioLoader1()

    iFold = 1
    path = BioLoader1.getPathIFold(iFold)
    bioloader.createTrainTestGraph(path)
    print(bioloader.graphData.is_undirected())
