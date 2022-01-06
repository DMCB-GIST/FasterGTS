import torch
from sklearn.model_selection import train_test_split
import csv
import numpy as np

def read_object_property_file(path, delimiter=',', cols_to_read=[0, 1],
                              keep_header=False):
    f = open(path, 'r')
    reader = csv.reader(f, delimiter=delimiter)
    data_full = np.array(list(reader))
    if keep_header:
        start_position = 0
    else:
        start_position = 1
    assert len(data_full) > start_position
    data = [[] for _ in range(len(cols_to_read))]
    for i in range(len(cols_to_read)):
        col = cols_to_read[i]
        data[i] = data_full[start_position:, col]
    f.close()
    if len(cols_to_read) == 1:
        data = data[0]
    return data

delimiter='\t'
cols_to_read=[0]
keep_header=True

data_path = '/NAS_Storage1/leo8544/CanDIS/data/ChEMBL/chembl_22_clean_1576904_sorted_std_final.smi'

data = read_object_property_file(data_path,delimiter=delimiter,
                          cols_to_read=cols_to_read,keep_header=keep_header)


Max_atoms = 100
smiles =[ i for i in data if len(i) < Max_atoms ]


smiles, vsmiles = train_test_split(smiles, test_size=0.01, random_state=42)
    
max_len = Max_atoms
batch_size = 256

dataset_loader = torch.utils.data.DataLoader(
            dataset=smiles, batch_size=batch_size, shuffle=True, num_workers=0
        )