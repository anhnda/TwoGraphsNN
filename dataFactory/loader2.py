import config
from utils import utils
import numpy as np
from dataFactory import loadingMap

import torch
from torch_geometric.data import Data

from torch_geometric.utils import to_undirected


class BioLoader2:
    def __init__(self):
        self.drugInchiKey2Id = dict()
        # self.sideEffectName2Id = dict()
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

    def loadDrugName2Info(self):
        f = open(config.DRUGBANK_ATC_INCHI)
        dDrugName2Inchi = dict()
        dDrugName2SMILE = dict()
        while True:
            line = f.readline()
            if line == "":
                break
            parts = line.strip().split("\t")
            dDrugName2Inchi[parts[1].lower()] = parts[-1]
            dDrugName2SMILE[parts[1].lower()] = parts[4]
        f.close()
        self.drugName2Inchi = dDrugName2Inchi
        self.drugInchi2Name = utils.reverse_dict(dDrugName2Inchi)
        self.drugName2SMILE = dDrugName2SMILE
        return dDrugName2Inchi

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
                fingerPrint = np.zeros(config.CHEM_FINGERPRINT_SIZE, dtype=int)
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
        for k in inchiKeyList:
            v = dInchi2Features[k]
            chemId = utils.get_update_dict_index(self.drugInchiKey2Id, k)
            self.drugId2Features[chemId] = v
            nonZeros = np.nonzero(v)[0]
            # if k == "QTGIAADRBBLJGA-UHFFFAOYSA-N":
            #    print nonZeros
            for bit in nonZeros:
                vInchis = utils.get_insert_key_dict(dBit2Inchi, bit, [])
                vInchis.append(k)

        self.drugId2Inchikey = utils.reverse_dict(self.drugInchiKey2Id)

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

                ar = np.zeros(self.nProtein, dtype=int)
                for vv in proteinIdList:
                    ar[vv] = 1
                self.drugId2ProteinIndices[drugId] = ar

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
                    seId = utils.get_dict(self.dSeName2Id, seSider, -1)
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
            seName = self.dseId2Names[seId]
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
        seIds = set()
        # Load leaves:
        f = open(config.SE_ONTOLOTY_FILE_1)
        while True:
            line = f.readline()
            if line == "":
                break

            line = line.strip()
            parts = line.split("|")
            childId = parts[0]
            parentIds = parts[1].split(",")
            if childId not in selectedSeMeddraSet:
                continue
            seIds.add(childId)
            meddraSet.add(childId)
            currentParentSet = utils.get_insert_key_dict(parentDict, childId, set())
            for parentId in parentIds:
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
                childId = parts[0]
                if childId not in lv1MeddraSet:
                    continue
                parentIds = parts[1].split(",")
                currentParentSet = utils.get_insert_key_dict(parentDict, childId, set())
                for parentId in parentIds:
                    currentParentSet.add(parentId)
                    meddraSet.add(parentId)

            f.close()

        print(len(meddraSet), len(seIds), len(parentDict))

        self.dMeddra2Id = dict()
        sortedSeIds = sorted(list(seIds))
        sortedMeddraSet = sorted(list(meddraSet))

        for meddraId in sortedSeIds:
            utils.get_update_dict_index(self.dMeddra2Id, meddraId)
        for meddraId in sortedMeddraSet:
            if meddraId not in seIds:
                utils.get_update_dict_index(self.dMeddra2Id, meddraId)

        self.dSeName2Id = dict()

        for k, v in self.seName2MeddraSelected.items():
            seId = self.dMeddra2Id[v]
            self.dSeName2Id[k] = seId

        self.meddraSet = meddraSet
        self.seMeddraIds = seIds
        self.seParentDict = parentDict
        self.dseId2Names = utils.reverse_dict(self.dSeName2Id)

    def loadPPI(self):
        ppis = list()
        fin = open(config.PPI_FILE)
        while True:
            line = fin.readline()
            if line == "":
                break
            parts = line.strip().split("\t")
            ppis.append([parts[0], parts[1]])
        fin.close()
        self.ppi = ppis

    def createSharedGraph(self):
        NUM_NODES = 0

        # CHEM
        self.CHEM_OFFSET = NUM_NODES

        self.loadDrugChemFeatures()
        self.loadDrugName2Info()

        NUM_NODES += config.CHEM_FINGERPRINT_SIZE

        self.nDrug = len(self.drugInchiKey2Id)

        self.loadDrug2Proteins()
        self.PROTEIN_OFFSET = NUM_NODES
        NUM_NODES += self.nProtein

        print(len(self.drugId2ProteinIndices))
        self.loadPathWay()

        self.PATHWAY_OFFSET = NUM_NODES
        NUM_NODES += self.nPathway

        self.DRUG_OFFSET = NUM_NODES
        NUM_NODES += self.nDrug

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
        drug_edge_index = []

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
                        drug_edge_index.append([pathWayId + self.PATHWAY_OFFSET, proteinId + self.PROTEIN_OFFSET])
                        self.matProtein2Pathway[proteinId, pathWayId] = 1

        self.matDrugChem = np.zeros((self.nDrug, config.CHEM_FINGERPRINT_SIZE))
        # Drug 2 Chem:

        for drugId in range(self.nDrug):
            chems = self.drugId2Features[drugId]
            for chemId in chems:
                chemId = int(chemId)
                drug_edge_index.append([chemId + self.CHEM_OFFSET, drugId + self.DRUG_OFFSET])
                self.matDrugChem[drugId, chemId] = 1
        # Drug 2 Protein

        self.matDrugProtein = np.zeros((self.nDrug, self.nProtein))
        for drugId in range(self.nDrug):
            proteinIndices = self.drugId2ProteinIndices[drugId]
            for proteinId in proteinIndices:
                proteinId = int(proteinId)
                drug_edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                self.matDrugProtein[drugId, proteinId] = 1

        # Protein 2 Protein
        self.loadPPI()
        cc = 0
        for pp in self.ppi:
            p1, p2 = pp
            proteinId1 = utils.get_dict_index_only(self.protein2Id, p1)
            proteinId2 = utils.get_dict_index_only(self.protein2Id, p2)
            if proteinId1 != -1 and proteinId2 != -1:
                cc += 1
                drug_edge_index.append([proteinId1 + self.PROTEIN_OFFSET, proteinId2 + self.PROTEIN_OFFSET])
                drug_edge_index.append([proteinId2 + self.PROTEIN_OFFSET, proteinId1 + self.PROTEIN_OFFSET])
        print ("N Protein-Protein: ", cc)

        # Add DART
        if config.DART:
            self.loadDART()
            for k, v in self.dartBenchMark.items():
                drugId, _ = k
                proteinIds = v
                for proteinId in proteinIds:
                    proteinId = int(proteinId)
                    drug_edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                    self.matDrugProtein[drugId, proteinId] = 1

        # drugPathway:
        self.matDrugPathway = np.zeros((self.nDrug, self.nPathway))
        m = np.dot(self.matDrugProtein, self.matProtein2Pathway)
        m[m >= 1] = 1
        self.matDrugPathway = m

        # SE
        se_edge_index = []
        for child, parents in self.seParentDict.items():
            seChildId = utils.get_dict(self.dMeddra2Id, child, -1)
            for parent in parents:
                if parent == "":
                    continue
                seParentId = utils.get_dict(self.dMeddra2Id, parent, -1)
                if seParentId != -1:
                    se_edge_index.append([seParentId + self.SE_OFFSET, seChildId + self.SE_OFFSET])

        self.x = x
        self.drug_edge_index = drug_edge_index
        self.se_edge_index = se_edge_index

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
        print("Input Featuers: ", inputAll.shape)
        self.trainInpMat = inputAll[drugTrainIds, :]
        self.testInpMat = inputAll[drugTestIds, :]

    def createTrainTestGraph(self, fPath, allTrain=False):
        self.createSharedGraph()
        self.loadTrainTest(fPath, allTrain)
        # Add training drug-se edges:
        # for drugId in self.drugTrainIdList:
        #     seIds = self.drugId2Ses[drugId]
        #     for seId in seIds:
        #         self.edge_index.append([drugId + self.DRUG_OFFSET, seId + self.SE_OFFSET])
        #         self.edge_index.append([seId + self.SE_OFFSET, drugId + self.DRUG_OFFSET])

        drug_edge_index = torch.tensor(self.drug_edge_index, dtype=torch.long).t().contiguous()
        se_edge_index = torch.tensor(self.se_edge_index, dtype=torch.long).t().contiguous()

        if config.UN_DIRECTED:
            drug_edge_index = to_undirected(drug_edge_index)
            se_edge_index = to_undirected(se_edge_index)

        self.drugGraphData = Data(x=self.x, edge_index=drug_edge_index)
        self.seGraphData = Data(x=self.x, edge_index=se_edge_index)

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

        print("Undirected graph: ", self.drugGraphData.is_undirected(), "Config: ", config.UN_DIRECTED)

    @staticmethod
    def getTrainPathPref():
        pathTrainPref = "%s%s" % (config.PATH_KFOLD_ATC_INCHIKEY_SIDEEFFECT_BYDRUG, config.TRAIN_PREFIX)
        return pathTrainPref

    @staticmethod
    def getPathIFold(iFold):
        pathTrain = "%s%s" % (BioLoader2.getTrainPathPref(), iFold)
        return pathTrain


if __name__ == "__main__":
    bioloader2 = BioLoader2()

    iFold = 1
    path = BioLoader2.getPathIFold(iFold)
    bioloader2.createTrainTestGraph(path)
    print(bioloader2.drugGraphData.is_undirected(), bioloader2.seGraphData.is_undirected())
