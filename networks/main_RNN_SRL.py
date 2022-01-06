from rdkit import Chem
import math
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import random

from trainer import TrainerConfig, ChembleTrainer
from torch.cuda.amp import GradScaler

from sklearn.model_selection import train_test_split

from data import GeneratorData, MetadataGenerate, GeneratorData, ChembleSmileDataset

import deepchem as dc
from layers.graph import *
import keras

from sklearn.utils import shuffle

import random,os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"

current_gpu = 0


Max_atoms = 100
max_len = Max_atoms
n_characters = 45


gen_data_path = '/FasterGTS/data/ChEMBL/chembl_22_clean_1576904_sorted_std_final.smi'

Drug_info_file = '/FasterGTS/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '/FasterGTS/data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '/FasterGTS/data/GDSC/drug_graph_feat'
Genomic_mutation_file = '/FasterGTS/data/CCLE/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '/FasterGTS/data/CCLE/GDSC_IC50.csv'
Gene_expression_file = '/FasterGTS/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '/FasterGTS/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

try:
    print("already data ",gen_data)
except:
    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                             cols_to_read=[0], keep_header=True, tokens=tokens)



checkpoint = '/FasterGTS/pre_trained_weights/adj_best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP.h5'
predictor = keras.models.load_model(checkpoint,custom_objects={'GraphConv':GraphConv})
try:
    from deepchem.feat.adjacency_fingerprint import AdjacencyFingerprint
    adj_fp_fun  = AdjacencyFingerprint(max_n_atoms=100)
except:
    adj_fp_fun  = dc.feat.adjacency_fingerprint.AdjacencyFingerprint(max_n_atoms=100)
## all cell lines
mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,
                                                                                                Cell_line_info_file,
                                                                                                Genomic_mutation_file,
                                                                                                Drug_feature_file,
                                                                                                Gene_expression_file,
                                                                                                Methylation_file,False)



from stackRNN import StackAugmentedRNN

from personalized_RL_rnn import *

hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.0003
optimizer_instance = torch.optim.Adadelta

torch.cuda.set_device(current_gpu)

generator = StackAugmentedRNN(input_size=gen_data.n_characters,
                                     hidden_size=hidden_size,
                                     output_size=gen_data.n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=True,
                                     optimizer_instance=optimizer_instance, lr=lr,
                                     gpuNum=current_gpu)

device = torch.device("cuda:"+str(current_gpu))
generator.to(device)

train_cell_ic50 =  pd.read_csv('/FasterGTS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

NTBC_cells = pd.read_csv("/FasterGTS/NTBC_cells_info.csv",index_col = 0)

test_x_mutation = mutation_feature.loc[cell_lines]
test_x_mutation = np.array(test_x_mutation).reshape(test_x_mutation.shape[0],1,test_x_mutation.shape[1],1)

test_x_expr = gexpr_feature.loc[cell_lines]
test_x_expr =  np.array(test_x_expr)

test_x_methylation = methylation_feature.loc[cell_lines]
test_x_methylation = np.array(test_x_methylation)


alpha = 0.8
beta = 1.8

n_samples = 300
max_n_samples = 5000


from personalized_RL_rnn import generate_samples

for index in range(13,len(NTBC_cells)):

    target_cell = NTBC_cells.index[index]
    thres_ic = NTBC_cells['Thres_IC50'][index]
    thres_z = NTBC_cells['Thres_Z'][index]

    x_mutation = mutation_feature.loc[target_cell]
    x_mutation = np.array(x_mutation).reshape(1,1,x_mutation.shape[0],1)

    x_expr = gexpr_feature.loc[target_cell]
    x_expr = np.array(x_expr).reshape(1,x_expr.shape[0])

    x_methylation = methylation_feature.loc[target_cell]
    x_methylation = np.array(x_methylation).reshape(1,x_methylation.shape[0])

    path = '/FasterGTS/pre_trained_weights/SRL_rnn_'+target_cell+'.pt'
    generator.load_model(path)
    generator.to(device)

    samples = []

    while len(samples) <= max_n_samples:
        samples += generate_samples(generator,gen_data,n_samples)
        samples = list(set(samples))
        print(len(samples))

    valid_samples = []

    for smi in samples:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            adj_fp_data =  adj_fp_fun.featurize([mol])
            single_adj_data = adj_fp_data[0][0].reshape(1,100,100)
            single_drug_feat_data = adj_fp_data[0][1].reshape(1,100,75)
            adj_data = np.tile(single_adj_data,[x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, x_mutation,x_expr,x_methylation]

            IC50 = predictor.predict(input_data)
            IC50 = np.average(list(IC50))

            adj_data = np.tile(single_adj_data,[test_x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[test_x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, test_x_mutation,test_x_expr,test_x_methylation]

            test_pred_value = predictor.predict(input_data)
            test_pred_value = test_pred_value.reshape(test_pred_value.shape[0])
            mean = np.mean(test_pred_value)
            std = np.std(test_pred_value)
            z_score = round((IC50-mean)/std,2)
            reward = 1.

            if z_score <= thres_z and IC50 <= thres_ic:
                reward = np.exp(alpha*(-z_score+thres_z))+beta*np.log(-IC50+thres_ic+1)

            print("z_score ",z_score)
            print("IC50 ",IC50)
            print('reward', reward)

            valid_samples.append([smi,round(IC50,4),round(z_score,4),round(reward,4)])

    pd.DataFrame(data = valid_samples,columns=['sample','IC50','z-score','reward']).\
            sort_values(by=['reward'],ascending=False).\
            to_csv('/FasterGTS/SRL_rnn_results/SRL_rnn_'+target_cell+'.csv')
