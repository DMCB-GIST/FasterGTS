"""
This code is modified and added by Sejin Park, based on ReLeaSE (Reinforcement Learning for Structural Evolution)
and https://github.com/kimmo1019/DeepCDR, DeepCDR (a hybrid graph convolutional network for predicting cancer drug response).

The original codes are from https://github.com/isayev/ReLeaSE
and https://github.com/kimmo1019/DeepCDR, DeepCDR (a hybrid graph convolutional network for predicting cancer drug response).

Information about thier copyrights are in https://github.com/isayev/ReLeaSE/blob/master/LICENSE
and https://github.com/kimmo1019/DeepCDR/blob/master/LICENSE.

"""

import random,os,sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
import pickle
from rdkit import Chem, DataStructs

from utils import canonical_smiles

import matplotlib.pyplot as plt
import seaborn as sns

from data import MetadataGenerate, GeneratorData

from reinforcement import Reinforcement
from stackRNN import StackAugmentedRNN

import pandas as pd
from datetime import datetime

import deepchem as dc
from rdkit import Chem
from layers.graph import *
import keras
import csv
import hickle as hkl


from sklearn.utils import shuffle


gen_data_path = '/FasterGTS/data/ChEMBL/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
          '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
          '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n']

TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD_READ']
Drug_info_file = '/FasterGTS/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '/FasterGTS/data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '/FasterGTS/data/GDSC/drug_graph_feat'
Genomic_mutation_file = '/FasterGTS/data/CCLE/genomic_mutation_34673_demap_features.csv'
Cancer_response_exp_file = '/FasterGTS/data/CCLE/GDSC_IC50.csv'
Gene_expression_file = '/FasterGTS/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '/FasterGTS/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'
Max_atoms = 100


import random,os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

current_gpu = 3

try:
    print("already data ",gen_data)
except:
    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                             cols_to_read=[0], keep_header=True, tokens=tokens)


def plot_hist(prediction, n_to_generate):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel='Predicted ln_IC50 Mean: '+str(prediction.mean())+' valid ratio: '+str( len(prediction)/n_to_generate) ,
           title='Distribution of predicted pIC50 for generated molecules')
    plt.show()

def estimate_and_update(generator, predictor, n_to_generate,adj_fp_fun,x_mutation,x_expr,x_methylation, **kwargs):
    generated = []
    pbar = tqdm(range(n_to_generate))
    for i in pbar:
        pbar.set_description("Generating molecules...")
        generated.append(generator.evaluate(gen_data, predict_len=120)[1:-1])

    sanitized = canonical_smiles(generated, sanitize=False, throw_warning=False)[:-1]
    unique_smiles = list(np.unique(sanitized))[1:]

    valid_smiles  = []
    mol_list = []
    for smiles in unique_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol and len(smiles) > 2:
            valid_smiles.append(smiles)
            mol_list.append(mol)
    mu_list = []
    expr_list = []
    meth_list = []
    for i in range(len(valid_smiles)):
        rand_index = random.randrange(0,x_mutation.shape[0])
        rand_mutation =  x_mutation[rand_index].reshape(1,x_mutation.shape[1],x_mutation.shape[2],x_mutation.shape[3])
        rand_expr =  x_expr[rand_index].reshape(1,x_expr.shape[1])
        rand_methylation =  x_methylation[rand_index].reshape(1,x_methylation.shape[1])
        mu_list.append(rand_mutation)
        expr_list.append(rand_expr)
        meth_list.append(rand_methylation)

    mu_list =  np.concatenate(mu_list,axis=0)
    expr_list =  np.concatenate(expr_list,axis=0)
    meth_list =  np.concatenate(meth_list,axis=0)


    adj_fp_list =  adj_fp_fun.featurize(mol_list)
    adj_data_list = []
    drug_feat_list = []
    for adj_fp in adj_fp_list:
        adj_data_list.append(np.tile(adj_fp[0][:,:],[1,1,1]))
        drug_feat_list.append(np.tile(adj_fp[1][:,:],[1,1,1]))

    drug_feat_data = np.concatenate(drug_feat_list,axis=0)
    adj_data = np.concatenate(adj_data_list,axis=0)

    input_data = [drug_feat_data,adj_data,mu_list,expr_list,meth_list]

    #smiles, prediction, nan_smiles = predictor.predict(unique_smiles)
    prediction = predictor.predict(input_data)
    prediction = prediction.reshape(prediction.shape[0])
    plot_hist(prediction, n_to_generate)

    return valid_smiles, prediction


def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def get_reward_max(smiles, predictor,adj_fp_fun,x_mutation,x_expr,x_methylation,threshold=0,invalid_reward=0.0):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        adj_fp_data =  adj_fp_fun.featurize([mol])
        adj_data = adj_fp_data[0][0].reshape(1,100,100)
        drug_feat_data = adj_fp_data[0][1].reshape(1,100,75)
        rand_index = random.randrange(0,x_mutation.shape[0])
        rand_mutation =  x_mutation[rand_index].reshape(1,x_mutation.shape[1],x_mutation.shape[2],x_mutation.shape[3])
        rand_expr =  x_expr[rand_index].reshape(1,x_expr.shape[1])
        rand_methylation =  x_methylation[rand_index].reshape(1,x_methylation.shape[1])

        input_data = [drug_feat_data,adj_data, rand_mutation,rand_expr,rand_methylation]
        pred_value = predictor.predict(input_data)[0][0]
        reward = 1
        if pred_value <threshold:
            #pred_value = -2
            reward = 10

        print("IC50: ",pred_value)
        #reward = np.exp(-pred_value/2)
        return reward

    return invalid_reward


checkpoint = '/FasterGTS/pre_trained_weights/adj_best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP.h5'
my_predictor = keras.models.load_model(checkpoint,custom_objects={'GraphConv':GraphConv})

adj_fp_fun  = dc.feat.adjacency_fingerprints.AdjacencyFingerprint(max_n_atoms=100)

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

train_cell_ic50 =  pd.read_csv('/FasterGTS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

train_x_mutation = mutation_feature.loc[cell_lines]
train_x_mutation = np.array(train_x_mutation).reshape(train_x_mutation.shape[0],1,train_x_mutation.shape[1],1)

train_x_expr = gexpr_feature.loc[cell_lines]
train_x_expr =  np.array(train_x_expr)

train_x_methylation = methylation_feature.loc[cell_lines]
train_x_methylation = np.array(train_x_methylation)

from stackRNN import StackAugmentedRNN

hidden_size = 1500
stack_width = 1500
stack_depth = 200
layer_type = 'GRU'
lr = 0.0001
optimizer_instance = torch.optim.Adadelta


current_gpu = 3
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

GD_space_network_path = '/FasterGTS/pre_trained_weights/GD_space_network_weight'
generator.load_model(GD_space_network_path)
device = torch.device("cuda:"+str(current_gpu))
generator.to(device)

from reinforcement import Reinforcement
threshold = 1
RL_max = Reinforcement(generator, my_predictor, get_reward_max,adj_fp_fun,
                       x_mutation=train_x_mutation,x_expr=train_x_expr,x_methylation=train_x_methylation)

rewards_max = []
rl_losses_max = []

# Setting up some parameters for the experiment
n_to_generate = 200
n_policy = 30

for i in range(n_to_generate):
    for j in range(n_policy):
        cur_reward, cur_loss = RL_max.policy_gradient(gen_data,n_batch=20)
        rewards_max.append(simple_moving_average(rewards_max, cur_reward))
        rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))

    plt.plot(rewards_max)
    plt.xlabel('Training iteration')
    plt.ylabel('Average reward')
    plt.show()

    plt.plot(rl_losses_max)
    plt.xlabel('Training iteration')
    plt.ylabel('Loss')
    plt.show()

    smiles_cur, prediction_cur = estimate_and_update(generator,
                                                     my_predictor,
                                                     1000,
                                                     adj_fp_fun,
                                                     train_x_mutation,
                                                     train_x_expr,
                                                     train_x_methylation)

    smiles_cur = shuffle(smiles_cur)
    print('Sample trajectories:')
    for sm in smiles_cur[:30]:
        print(sm)


torch.save(generator.state_dict(), '/FasterGTS/pre_trained_weights/rnn_rl_lr_e04_ep_187.pt')
