import torch
import numpy as np
from pysmiles import read_smiles

smiles = '[H][C@@]12CC[C@](O)(C(=O)CO)[C@@]1(C)C[C@H](O)[C@@]1(F)[C@@]2([H])CCC2=CC(=O)CC[C@]12C'
mol = read_smiles(smiles)

print (mol)
nodes = mol._node
keys = nodes.keys()
v = sorted(keys)
print (v)