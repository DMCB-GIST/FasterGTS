"""
This code is written by authors of FasterGTS.

All rights are reserved.
"""
from utils import get_samples
from rdkit import Chem
import math
from model import GPTConfig, ChemGPT
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

current_gpu = 2

Max_atoms = 101
max_len = Max_atoms
n_characters = 46

Drug_info_file = './FasterGTS/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = './FasterGTS/data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = './FasterGTS/data/GDSC/drug_graph_feat'
Genomic_mutation_file = './FasterGTS/data/CCLE/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = './FasterGTS/data/CCLE/GDSC_IC50.csv'
Gene_expression_file = './FasterGTS/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = './FasterGTS/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', '$']

chars = sorted(list(set(tokens)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


checkpoint = './FasterGTS/pre_trained_weights/adj_best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP.h5'
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

train_cell_ic50 =  pd.read_csv('./FasterGTS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

TNBC_cells = pd.read_csv("./FasterGTS/TNBC_cells_info.csv",index_col = 0)

test_x_mutation = mutation_feature.loc[cell_lines]
test_x_mutation = np.array(test_x_mutation).reshape(test_x_mutation.shape[0],1,test_x_mutation.shape[1],1)

test_x_expr = gexpr_feature.loc[cell_lines]
test_x_expr =  np.array(test_x_expr)

test_x_methylation = methylation_feature.loc[cell_lines]
test_x_methylation = np.array(test_x_methylation)

alpha = 0.8
beta = 1.8

device = torch.device("cuda:"+str(current_gpu))


from model import ChemGPT

mconf = GPTConfig(n_characters, Max_atoms,
	               n_layer=4, n_head=8, n_embd=256)

generator = ChemGPT(mconf).to(device)
trainable_generator = ChemGPT(mconf).to(device)

GD_space_network_path = './FasterGTS/pre_trained_weights/gpt_4layer.pt'

from mcts import *

constant = round(math.sqrt(8),2)
indices = list(range(0,21))
learning_rate = 6e-5
batch_size = 128
max_len = 100

if True:
    for index in indices:

        mode = 'lr_6e5_stop_10000_const_'+str(constant)

        target_cell = TNBC_cells.index[index]
        thres_ic = TNBC_cells['Thres_IC50'][index]
        thres_z = TNBC_cells['Thres_Z'][index]

        x_mutation = mutation_feature.loc[target_cell]
        x_mutation = np.array(x_mutation).reshape(1,1,x_mutation.shape[0],1)

        x_expr = gexpr_feature.loc[target_cell]
        x_expr = np.array(x_expr).reshape(1,x_expr.shape[0])

        x_methylation = methylation_feature.loc[target_cell]
        x_methylation = np.array(x_methylation).reshape(1,x_methylation.shape[0])


        img_save_dir = './FasterGTS/FasterGTS_Trainable_gpt_imgs/'+mode+'_'+target_cell+'_'
        file_save_dir = './FasterGTS/FasterGTS_Trainable_gpt_results/'+mode+'_'+target_cell+'_'
        dict_save_dir = './FasterGTS/FasterGTS_Trainable_gpt_dict/'+mode+'_'+target_cell+'_'

        config = TrainerConfig(max_epochs= max_epochs, batch_size=batch_size, learning_rate=learning_rate,
    	                      lr_decay=True, warmup_tokens=0.1*1024*max_len,
                              final_tokens=max_epochs*1024*max_len, num_workers=10,
                              t_path = './FasterGTS/pre_trained_weights/gpt_4layer_'+mode+'_'+target_cell+'.pt')

        optimizer = trainable_generator.configure_optimizers(config)
        scaler = GradScaler()

        generator.load_model(GD_space_network_path)
        trainable_generator.load_model(GD_space_network_path)

        generator.eval()

        state = State_trainable_GPT(generator,trainable_generator, predictor, adj_fp_fun,stoi,itos,
                        mutation_feature,gexpr_feature,methylation_feature,
                        cell_lines,target_cell,thres_ic = thres_ic,thres_z = thres_z,
                        #numSample = 20 -> for wo GA
                        numSample = 10, alpha = 0.8, beta = 1.8, generation_ratio = 0.7,
                        train_batch=128, max_queue=1024, mutation_rate = 0.01, num_GA =60,
                        optimizer = optimizer, scaler = scaler)

        my_mcts = MCTS(explorationConstant = constant, State = state,
                       img_save_dir=img_save_dir,file_save_dir=file_save_dir,
                        self_train = True, GA = True, trainable = True)

        my_mcts.simulate(start=my_mcts.root.nth, end = 500, stop = 10000,
                         numActions=5,dict_save_dir = dict_save_dir, mode = mode)

        file_name = '_'+str(my_mcts.root.nth)+'th_final_output.csv'
        my_mcts.save_molecule_list(my_mcts.file_save_dir+file_name)

        trainable_generator.save_model('.//FasterGTS_gpt_weight/'+mode+'_'+target_cell+'_gpt.pt')
