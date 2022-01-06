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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

indices_list = []
indices_list.append([0,1,2,3,4])
indices_list.append([5,6,7,8])
indices_list.append([9,10,11,12])
indices_list.append([13,14,15,16])
indices_list.append([17,18,19,20])

current_gpu = 4
indices = indices_list[current_gpu]

Max_atoms = 101
max_len = Max_atoms
n_characters = 46

Drug_info_file = '/FasterGTS/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '/FasterGTS/data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '/FasterGTS/data/GDSC/drug_graph_feat'
Genomic_mutation_file = '/FasterGTS/data/CCLE/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '/FasterGTS/data/CCLE/GDSC_IC50.csv'
Gene_expression_file = '/FasterGTS/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '/FasterGTS/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', '$']

chars = sorted(list(set(tokens)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }


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



device = torch.device("cuda:"+str(current_gpu))


from SRL_gpt import *
from model import ChemGPT

mconf = GPTConfig(n_characters, Max_atoms,
	               n_layer=4, n_head=8, n_embd=256)
generator = ChemGPT(mconf).to(device)


RL_path = '/FasterGTS/pre_trained_weights/gpt_4layer_rl_lr_5e05_ep_400.pt'


train_cell_ic50 =  pd.read_csv('/FasterGTS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

TNBC_cells = pd.read_csv("/FasterGTS/TNBC_cells_info.csv",index_col = 0)

test_x_mutation = mutation_feature.loc[cell_lines]
test_x_mutation = np.array(test_x_mutation).reshape(test_x_mutation.shape[0],1,test_x_mutation.shape[1],1)

test_x_expr = gexpr_feature.loc[cell_lines]
test_x_expr =  np.array(test_x_expr)

test_x_methylation = methylation_feature.loc[cell_lines]
test_x_methylation = np.array(test_x_methylation)


alpha = 0.8
beta = 1.8

n_to_generate = 20
n_policy = 500
n_try = 1000


for index in indices:
    generator.load_state_dict(torch.load(RL_path))
    config = TrainerConfig(max_epochs=n_policy, batch_size=n_to_generate, learning_rate=5e-5,
	                      lr_decay=True, warmup_tokens=0.1*n_to_generate*n_policy*max_len,
                          final_tokens=n_policy*n_to_generate*max_len,
	                      num_workers=10, ckpt_path = './weights/gpt_4_SRL.pt')

    scaler = GradScaler()
    optimizer = generator.configure_optimizers(config)


    target_cell = TNBC_cells.index[index]
    thres_ic = TNBC_cells['Thres_IC50'][index]
    thres_z = TNBC_cells['Thres_Z'][index]

    x_mutation = mutation_feature.loc[target_cell]
    x_mutation = np.array(x_mutation).reshape(1,1,x_mutation.shape[0],1)

    x_expr = gexpr_feature.loc[target_cell]
    x_expr = np.array(x_expr).reshape(1,x_expr.shape[0])

    x_methylation = methylation_feature.loc[target_cell]
    x_methylation = np.array(x_methylation).reshape(1,x_methylation.shape[0])


    img_save_dir = '/FasterGTS/PRL_gpt_imgs/'
    save_dir = '/FasterGTS/PRL_gpt_results/'+target_cell+'_PRL_gpt_results.csv'
    rate_save_dir= '/FasterGTS/PRL_gpt_rate/'

    RL_max = Personalized_RL(generator, predictor, adj_fp_fun, stoi=stoi,
                         x_mutation=x_mutation,x_expr=x_expr,x_methylation=x_methylation,
                         test_x_mutation=test_x_mutation,test_x_expr=test_x_expr,test_x_methylation=test_x_methylation,
                         alpha=alpha,beta=beta,thres_ic=thres_ic,thres_z=thres_z)

    rewards_max, rl_losses_max = train(RL_max, generator,target_cell,optimizer,scaler,device,itos,stoi,
                                       n_to_generate,n_policy,n_try,save_dir, img_save_dir)
