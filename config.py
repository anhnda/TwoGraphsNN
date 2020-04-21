import os
from torch_geometric.nn import GCNConv, GraphConv, SAGEConv, GatedGraphConv, GATConv
from models.g3n.nn import XGAT, XSAGE
C_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR = "%s/data" % C_DIR
OUTPUT_DIR = "%s/out" % C_DIR
FIG_DIR = "%s/figs" % C_DIR

SAVEMODEL_DIR = "%s/saveModel" % C_DIR
KFOLD_DRUG_DIR = "%s/NTimeKFold" % DATA_DIR
KFOLD_BINARY_DIR = "%s/Binary" % KFOLD_DRUG_DIR

SE_ONTOLOTY_FILE_1 = "%s/SIDER/ONTO/Ontology_1.txt" % DATA_DIR
SE_ONTOLOGY_PARENT_FILE_1 = "%s/SIDER/ONTO/OntologyParent.txt_1" % DATA_DIR
SE_ONTOLOGY_PARENT_FILE_2 = "%s/SIDER/ONTO/OntologyParent.txt_2" % DATA_DIR
SE_MEDDRA_NAME_FILE = "%s/SIDER/ONTO/se_meddra.txt" % DATA_DIR
SE_NAME_PROP_FILE = "%s/SIDER/ONTO/SE_Names.dat.txt" % DATA_DIR

KEGG_PATHWAY_FILE = "%s/KEGG/path:hsa.txt" % DATA_DIR
KEGG_UNIPROT_PATHWAY_MAPPING_PATH = "%s/KEGG/uniprot_2_pathway.txt" % DATA_DIR

PUBCHEM_FILE = "%s/PubChem/Inchikey2ProfileFilled.dat" % DATA_DIR
PUBCHEM_FINGERPRINT_FILE = "%s/PubChem/pubchem_fingerprints.txt" % DATA_DIR
INCHI_MORGAN_FILE = "%s/PubChem/InchikeysMorgan.txt" % DATA_DIR

PPI_FILE = "%s/HPRD/PPI_UNIPROT.txt" % DATA_DIR

SIDER_ATC_SIDEFFECT = "%s/SIDER/atc_side_effect_Filter" % DATA_DIR
DRUGBANK_ATC_INCHI = "%s/DrugBank/ATC_2_Inchikeys.txt" % DATA_DIR

SMILE2GRAPH = "%s/DrugBank/SMILDE2GRAPH.dat" % DATA_DIR

DART_RAW_FILE = "%s/DART/DART.dat" % DATA_DIR
DART_SE_MATCH_FILE = "%s/DART/DART_SE_MATCH.dat" % DATA_DIR
DART_DRUG_MATCH_FILE = "%s/DART/DART_DRUG_MATCH.dat" % DATA_DIR

BINDINGDB_EXTENDED = "%s/DrugBank/BindingDTB" % DATA_DIR
DRUGBANK_INCHI_PROTEIN = "%s/DrugBank/DRUBBANK_INCHIKEY_2_PROTEIN.dat" % DATA_DIR

PATH_KFOLD_ATC_INCHIKEY_SIDEEFFECT_BYDRUG = "%s/ATCInchikeySideEffectByDrug.txt" % KFOLD_DRUG_DIR
TRAIN_PREFIX = "_train_"
P3_PREFIX = "_P3_"

TEST_PREFIX = "_test_"

LATENT_GROUP_PATTERN_PATH_PREFIX = "%s/LatentGroup_Fold_" % OUTPUT_DIR
TOP_DESCRIPTORS_PATH_PREFIX = "%s/TopDescriptors_Fold_" % OUTPUT_DIR

INSIDE_PATTERN_FNAME = "INSIDE_Patterns.dat"
INSIDE_DRUGMAT_FNAME = "INSIDE_DrugMat.dat"
INSIDE_SEMAT_FNAME = "INSIDE_SeMat.dat"

INSIDE_FULL_SUFFIX = "_FULL"

FILTERED_SE_FNAME = "SE_Names.dat"

T_DRUG = 0
T_PROTEIN = 1
T_PATHWAY = 2
T_SIDEEFFECT = 3
T_NONE = -1

DICT_SPARSE = 0
ALL_SPARSE = 1
ALL_DENSE = 2
STYPE_MAP = {0: 'DICT_SPARSE', 1: 'ALL_SPARSE', 2: 'DENSE'}

N_TOP_GROUP_FEATURE = 20
MAX_TOP_GROUP_FEATURE = 25
ALPHA_SMOOTH = 5

TRAIN_D_PREFIX = "outputTrainP_"
TEST_D_PREFIX = "outputTestP_"
SIM_D_PREFIX = "simTrainTestIndices_"
SIM_D_W_PREFIX = "simTrainTestWeights_"

SKIP_INCHI = ["JLKIGFTWXXRPMT-UHFFFAOYSA-N", "JYGXADMDTFJGBT-VWUMJDOOSA-N"]
TORCH_SEED = int('1101001101010011010110101010101', 2)

PPI = True
UN_DIRECTED = True
DART = True
DART_ADD = True
BOTH_CHEM = False
MORGAN = False

SE_PROP_THRES = 0.1

N_MORGAN = 2048
N_PUBCHEM = 888  # 7 ending bits are padding.
MAX_NODE = 8000

if BOTH_CHEM:
    CHEM_FINGERPRINT_SIZE = N_MORGAN + N_PUBCHEM
elif MORGAN:
    CHEM_FINGERPRINT_SIZE = N_MORGAN
else:
    CHEM_FINGERPRINT_SIZE = N_PUBCHEM

IFOLD = 1
NTIMES_KFOLD = 5
K_FOLD = 10

EMBED_DIM = 100
N_EPOCH = 161

OPTIMIZER = "Adam"

N_LAYER_LEVEL_2 = 3
LEVEL_2_LAYER = XSAGE

N_LAYER_LEVEL_1 = 3
LEVEL_1_LAYER = SAGEConv
