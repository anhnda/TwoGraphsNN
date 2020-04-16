import config
from utils import utils
import numpy as np
from dataFactory import loadingMap, MoleculeFactory2

import torch
from torch_geometric.data import Data

from torch_geometric.utils import to_undirected


class BioLoader5P2:

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

        self.moleculeFactory = MoleculeFactory2.ModeculeFactory2()

    def loadValidInchi(self):
        path = BioLoader5P2.getPathNTIMESIFold(0, 0)
        fin = open(path)
        self.validInchi = set()
        while True:
            line = fin.readline()
            if line == "":
                break
            self.validInchi.add(line.split("|")[0])

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
                fingerPrint = np.zeros(config.N_MORGAN, dtype=int)
                for i, v in enumerate(morganString):
                    if float(v) == 1:
                        fingerPrint[i] = 1
                d[inchi] = fingerPrint
            fin.close()
            return d

        if config.BOTH_CHEM:
            dInchi2Morgan = loadMorgan()
            dInchi2PubChem = utils.load_obj(config.PUBCHEM_FILE)
            dInchi2Features = dict()
            for k, v in dInchi2Morgan.items():
                vPubchem = dInchi2PubChem[k]
                v = np.concatenate((v, vPubchem))
                dInchi2Features[k] = v
        elif config.MORGAN:
            dInchi2Features = loadMorgan()
        else:
            dInchi2Features = utils.load_obj(config.PUBCHEM_FILE)
        inchiKeyList = sorted(list(dInchi2Features.keys()))
        dBit2Inchi = dict()

        self.loadValidInchi()
        for k in inchiKeyList:
            if k in config.SKIP_INCHI:
                continue
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

        self.drugId2Inchikey = utils.reverse_dict(self.drugInchiKey2Id)
        self.drugId2SMILE = dict()
        for drugId in range(len(self.drugId2Inchikey)):
            inchi = self.drugId2Inchikey[drugId]
            name = self.drugInchi2Name[inchi]
            smile = self.drugName2SMILE[name]
            self.drugId2SMILE[drugId] = smile
            self.moleculeFactory.addSMILE(smile)

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
            seName = self.dSeId2Name[seId]
            proList = []
            for p in v:
                proList.append(self.id2ProteinName[p])
            fout.write("%s|%s|%s\n" % (drugName, seName, ",".join(proList)))
        fout.close()
        print("DART Info: Drug, Se, Protein, Pair: ", len(drugSetX), len(seSetX), len(proteinSetX), len(dDartBenchMark))
        return dDartBenchMark



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

    def loadSideEffect(self):

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
        fin.close()

        print("Num selected se: ", len(selectedSeNames), "Thres: ", config.SE_PROP_THRES)
        dSeName2Id = dict()
        selectedSeNames = sorted(list(selectedSeNames))
        for seId, seName in enumerate(selectedSeNames):
            dSeName2Id[seName] = seId

        self.dSeName2Id = dSeName2Id
        self.dSeId2Name = utils.reverse_dict(self.dSeName2Id)
        self.nSe = len(dSeName2Id)



    def createEdges(self):

        # Load Side effect for DART
        self.loadSideEffect()

        # Drug Features
        self.loadDrugName2Info()
        self.loadDrugChemFeatures()
        self.nDrug = len(self.drugInchiKey2Id)

        NUM_NODES = 0

        self.DRUG_OFFSET = NUM_NODES
        NUM_NODES += self.nDrug

        # PROTEIN
        self.loadDrug2Proteins()
        self.PROTEIN_OFFSET = NUM_NODES
        NUM_NODES += self.nProtein

        print(len(self.drugId2ProteinIndices))
        self.loadPathWay()

        self.PATHWAY_OFFSET = NUM_NODES
        NUM_NODES += self.nPathway

        self.ATOM_OFFSET = NUM_NODES
        NUM_NODES += self.moleculeFactory.getNumAtom()
        #
        self.graphBatch = self.moleculeFactory.createBatchGraph(self.ATOM_OFFSET)
        self.N_ATOMFEATURE = self.moleculeFactory.N_FEATURE

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
                        drug_edge_index.append([proteinId + self.PROTEIN_OFFSET, pathWayId + self.PATHWAY_OFFSET])
                        self.matProtein2Pathway[proteinId, pathWayId] = 1

        self.matDrugChem = np.zeros((self.nDrug, config.CHEM_FINGERPRINT_SIZE))
        # Drug 2 Chem Matrix Only:

        for drugId in range(self.nDrug):
            chems = self.drugId2Features[drugId]
            self.matDrugChem[drugId] = chems

        # Drug 2 Protein

        self.matDrugProtein = np.zeros((self.nDrug, self.nProtein))
        for drugId in range(self.nDrug):
            proteinIndices = self.drugId2ProteinIndices[drugId]

            for proteinId in proteinIndices:
                proteinId = int(proteinId)
                drug_edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                drug_edge_index.append([drugId + self.DRUG_OFFSET, proteinId + self.PROTEIN_OFFSET])

                self.matDrugProtein[drugId, proteinId] = 1

        # Protein 2 Protein
        cc = 0
        if config.PPI:
            self.loadPPI()
            for pp in self.ppi:
                p1, p2 = pp
                proteinId1 = utils.get_dict_index_only(self.protein2Id, p1)
                proteinId2 = utils.get_dict_index_only(self.protein2Id, p2)
                if proteinId1 != -1 and proteinId2 != -1:
                    cc += 1
                    drug_edge_index.append([proteinId1 + self.PROTEIN_OFFSET, proteinId2 + self.PROTEIN_OFFSET])
                    drug_edge_index.append([proteinId2 + self.PROTEIN_OFFSET, proteinId1 + self.PROTEIN_OFFSET])
            print("N Protein-Protein: ", cc)
        else:
            print("Skip Protein-Protein")

        # Add DART
        self.dartPair = set()
        if config.DART:
            self.loadDART()
            for k, v in self.dartBenchMark.items():
                drugId, seId = k
                self.dartPair.add((drugId, seId))
                proteinIds = v
                for proteinId in proteinIds:
                    proteinId = int(proteinId)
                    drug_edge_index.append([proteinId + self.PROTEIN_OFFSET, drugId + self.DRUG_OFFSET])
                    drug_edge_index.append([drugId + self.DRUG_OFFSET, proteinId + self.PROTEIN_OFFSET])
                    self.matDrugProtein[drugId, proteinId] = 1

            # Remake drugId2ProteinList
            drugId2ProteinList2 = dict()
            for drugId in range(self.nDrug):
                drug2ProteinAr = self.matDrugProtein[drugId, :]
                drug2ProteinLs = np.nonzero(drug2ProteinAr)[0]
                drugId2ProteinList2[drugId] = drug2ProteinLs

            self.drugId2ProteinIndices = drugId2ProteinList2

        # drugPathway:
        self.matDrugPathway = np.zeros((self.nDrug, self.nPathway))
        m = np.dot(self.matDrugProtein, self.matProtein2Pathway)
        m[m >= 1] = 1
        self.matDrugPathway = m

        self.x = x
        self.drug_edge_index = drug_edge_index
        self.drugFeatures = torch.from_numpy(self.matDrugChem).float()

        self.proteinNodeIds = self.x[self.PROTEIN_OFFSET: self.PROTEIN_OFFSET + self.nProtein]

        print("Drug-Protein: ", np.sum(self.matDrugProtein), "Protein-Protein: ", cc, "Protein-Pathways: ",
              np.sum(self.matProtein2Pathway))

    def loadTrainTest(self, fPath, allTrain=False):


        fin = open(fPath)
        dDrugId2SideEffect = dict()

        drugTrainIds = list()
        drugTestIds = list()
        drugValidateIds = list()

        currenDrugSet = drugTrainIds

        inchiSet = set()
        while True:
            line = fin.readline()
            if line == "":
                break
            line = line.strip()
            if line.startswith("#"):
                if allTrain == False:
                    if line.__contains__("Val"):
                        currenDrugSet = drugValidateIds
                    else:
                        currenDrugSet = drugTestIds
                continue
            parts = line.split("|")
            inchi = parts[0]

            # SKIP WRONG DATA:
            if inchi == "JLKIGFTWXXRPMT-UHFFFAOYSA-N" or inchi == "JYGXADMDTFJGBT-VWUMJDOOSA-N":
                continue
            sideEffects = parts[-1].split(",")
            drugId = utils.get_dict_index_only(self.drugInchiKey2Id, inchi)
            if drugId == -1:
                print("No Inchi: ", inchi)
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
        self.drugValidateIdList = drugValidateIds

        self.drugId2Ses = dDrugId2SideEffect

        print(len(drugTrainIds), len(drugTestIds), self.nSe)
        trainOutMatrix = np.zeros((len(drugTrainIds), self.nSe), dtype=float)
        rowId = 0
        for drugId in drugTrainIds:
            seIds = dDrugId2SideEffect[drugId]

            for seId in seIds:
                trainOutMatrix[rowId, seId] = 1
                if (drugId, seId) in self.dartPair:
                    trainOutMatrix[rowId, seId] = 1

            rowId += 1

        testOutMatrix = np.zeros((len(drugTestIds), self.nSe), dtype=float)
        rowId = 0
        for drugId in drugTestIds:
            seIds = dDrugId2SideEffect[drugId]
            for seId in seIds:
                testOutMatrix[rowId, seId] = 1
            rowId += 1

        validOutMatrix = np.zeros((len(drugValidateIds), self.nSe), dtype=float)

        rowId = 0
        for drugId in drugValidateIds:
            seIds = dDrugId2SideEffect[drugId]
            for seId in seIds:
                validOutMatrix[rowId, seId] = 1
            rowId += 1

        self.trainOutMatrix = torch.from_numpy(trainOutMatrix).float()
        self.testOutMatrix = torch.from_numpy(testOutMatrix).float()
        self.validOutMatrix = torch.from_numpy(validOutMatrix).float()

        inputAll = np.concatenate([self.matDrugChem, self.matDrugProtein, self.matDrugPathway], axis=1)
        self.trainInpMat = inputAll[drugTrainIds, :]
        self.testInpMat = inputAll[drugTestIds, :]
        self.validateInpMat = inputAll[drugValidateIds, :]

        print("Num Drug-SE: ", np.sum(trainOutMatrix) + np.sum(testOutMatrix))

    def createTrainTestVal(self, fPath, allTrain=False):
        self.createEdges()
        self.loadTrainTest(fPath, allTrain)

        drugTrainNodeIds = list()
        drugTestNodeIds = list()
        drugValidateNodeIds = list()

        nonTrainDrugNodeIds = set()
        for drugId in self.drugTrainIdList:
            drugTrainNodeIds.append(drugId + self.DRUG_OFFSET)
        for drugId in self.drugTestIdList:
            drugTestNodeIds.append(drugId + self.DRUG_OFFSET)
            nonTrainDrugNodeIds.add(drugId + self.DRUG_OFFSET)
        for drugId in self.drugValidateIdList:
            drugValidateNodeIds.append(drugId + self.DRUG_OFFSET)
            nonTrainDrugNodeIds.add(drugId + self.DRUG_OFFSET)

        drug_edge_index = torch.tensor(self.drug_edge_index, dtype=torch.long).t().contiguous()

        if config.UN_DIRECTED:
            drug_edge_index = to_undirected(drug_edge_index)

        self.drugGraphData = Data(x=self.x, edge_index=drug_edge_index)
        print("Num edges: ", self.drugGraphData.num_edges)

        self.drugTrainNodeIds = torch.from_numpy(np.asarray(drugTrainNodeIds, dtype=np.long)).long()
        self.drugTestNodeIds = torch.from_numpy(np.asarray(drugTestNodeIds, dtype=np.long)).long()
        self.drugValidateNodeIds = torch.from_numpy(np.asarray(drugValidateNodeIds, dtype=np.long)).long()

        print("Undirected graph: ", self.drugGraphData.is_undirected(), "Config: ", config.UN_DIRECTED)

    @staticmethod
    def getTrainPathPref():
        pathTrainPref = "%s%s" % (config.PATH_KFOLD_ATC_INCHIKEY_SIDEEFFECT_BYDRUG, config.P3_PREFIX)
        return pathTrainPref

    @staticmethod
    def getPathIFold(iFold):
        pathTrain = "%s%s" % (BioLoader5P2.getTrainPathPref(), iFold)
        return pathTrain

    @staticmethod
    def getPathNTIMESIFold(timeFold, iFold):
        pathTrain = "%s%s_%s" % (BioLoader5P2.getTrainPathPref(), timeFold, iFold)
        return pathTrain


if __name__ == "__main__":
    # bioloader2 = BioLoader5P2()
    #
    # iFold = 1
    # path = BioLoader5P2.getPathIFold(iFold)
    # bioloader2.createTrainTestGraph(path)
    # print(bioloader2.drugGraphData.is_undirected(), bioloader2.seGraphData.is_undirected())
    pass