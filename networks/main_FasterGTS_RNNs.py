"""
This code is written by authors of FasterGTS.

All rights are reserved.
"""

import random,os,sys
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from data import MetadataGenerate, GeneratorData
import pandas as pd
from anytree import Node, RenderTree
import math
import deepchem as dc
from rdkit import Chem
from layers.graph import *
import keras
import csv

save_dir = './result/'

Drug_info_file = './data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = './data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = './data/GDSC/drug_graph_feat'
Genomic_mutation_file = './data/CCLE/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = './data/CCLE/GDSC_IC50.csv'
Gene_expression_file = './data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = './data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'


Max_atoms = 100

data_path = './data/ChEMBL/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

current_gpu = 0

try:
    print("already data ",data)
except:
    data = GeneratorData(training_data_path=data_path, delimiter='\t',
                           cols_to_read=[0], keep_header=True, tokens=tokens)

try:
    from deepchem.feat.adjacency_fingerprint import AdjacencyFingerprint
    adj_fp_fun  = AdjacencyFingerprint(max_n_atoms=100)
except:
    adj_fp_fun  = dc.feat.adjacency_fingerprint.AdjacencyFingerprint(max_n_atoms=100)

n_characters = 45
hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.005

mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,False)

train_cell_ic50 =  pd.read_csv('./train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

checkpoint = './pre_trained_weights/adj_best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP.h5'
predictor = keras.models.load_model(checkpoint,custom_objects={'GraphConv':GraphConv})

optimizer_instance = torch.optim.Adadelta
optimizer_instance_for_train = torch.optim.Adadelta

from stackRNN import StackAugmentedRNN

torch.cuda.set_device(current_gpu)

generator = StackAugmentedRNN(input_size=n_characters,
                                     hidden_size=hidden_size,
                                     output_size=n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=True,
                                     optimizer_instance=optimizer_instance, lr=lr,
                                     gpuNum=current_gpu)

trainable_generator = StackAugmentedRNN(input_size=n_characters,
                                     hidden_size=hidden_size,
                                     output_size=n_characters,
                                     layer_type=layer_type,
                                     n_layers=1, is_bidirectional=False, has_stack=True,
                                     stack_width=stack_width, stack_depth=stack_depth,
                                     use_cuda=True,
                                     optimizer_instance=optimizer_instance_for_train, lr=lr,
                                     gpuNum=current_gpu)



GD_space_network_path = './pre_trained_weights/GD_space_network_weight'

mode = 'GA_trainalbe_lr_5e-4_stop_10000'

TNBC_cells = pd.read_csv("./TNBC_cells_info.csv",index_col = 0)

from mcts import *
constant = math.sqrt(8)

index_list = list(range(0,21))

for index in index_list:

    target_cell = TNBC_cells.index[index]
    thres_ic = TNBC_cells['Thres_IC50'][index]
    thres_z = TNBC_cells['Thres_Z'][index]

    img_save_dir ='./FasterGTS_Trainable_RNNs_img/'+mode+'_'+str(constant)+'_'+target_cell
    dict_save_dir = './FasterGTS_Trainable_RNNs_dict/'+mode+'_'+str(constant)+'_'+target_cell+\
        '_ic50_'+str(thres_ic)+'_Zscore_'+str(thres_z)
    file_save_dir = './FasterGTS_Trainable_RNNs_file/'+mode+'_'+str(constant)+'_'+target_cell+\
        '_ic50_'+str(thres_ic)+'_Zscore_'+str(thres_z)

    trainable_generator.load_model(GD_space_network_path)
    generator.load_model(GD_space_network_path)

    generator.eval()

    state = State_trainable_RNNs(generator, trainable_generator, predictor, adj_fp_fun,
                    mutation_feature,gexpr_feature,methylation_feature,
                    cell_lines,target_cell,thres_ic = thres_ic,thres_z = thres_z,
                    data=data,numSample = 30, alpha = 0.8, beta = 1.8, generation_ratio = 0.7,
                    train_batch=128, max_queue=1024, mutation_rate = 0.01, num_GA =60)

    my_mcts = MCTS(explorationConstant = constant, State = state,
                   img_save_dir=img_save_dir,file_save_dir=file_save_dir,
                   self_train = True, GA = True)

    my_mcts.simulate(start=my_mcts.root.nth, end = 500, stop = 10000,
                     numActions=10,dict_save_dir = dict_save_dir, mode = mode)

    file_name = '_'+str(my_mcts.root.nth)+'th_final_output.csv'
    my_mcts.save_molecule_list(my_mcts.file_save_dir+file_name)

    trainable_generator.save_model('./FasterGTS_RNN_weight/'+target_cell+'_'+mode+'_RNNs.pt')
