"""
This code is modified and added by Sejin Park, based on ReLeaSE (Reinforcement Learning for Structural Evolution)
and https://github.com/kimmo1019/DeepCDR, DeepCDR (a hybrid graph convolutional network for predicting cancer drug response).

The original codes are from https://github.com/isayev/ReLeaSE
and https://github.com/kimmo1019/DeepCDR, DeepCDR (a hybrid graph convolutional network for predicting cancer drug response).

Information about thier copyrights are in https://github.com/isayev/ReLeaSE/blob/master/LICENSE
and https://github.com/kimmo1019/DeepCDR/blob/master/LICENSE.

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
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"

current_gpu = 4

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



def plot_hist(prediction, n_to_generate):
    print("Mean value of predictions:", prediction.mean())
    print("Proportion of valid SMILES:", len(prediction)/n_to_generate)
    ax = sns.kdeplot(prediction, shade=True)
    ax.set(xlabel='Predicted ln_IC50 Mean: '+str(prediction.mean())+' valid ratio: '+str(len(prediction)/n_to_generate) ,
           title='Distribution of predicted pIC50 for generated molecules')
    plt.show()

def estimate_and_update(generator, predictor, n_to_generate,stoi,itos,
                        adj_fp_fun,x_mutation,x_expr,x_methylation, **kwargs):

    x = torch.tensor([stoi[context]], dtype=torch.long)[None,...].repeat(n_to_generate, 1).to(device)
    y = get_samples(generator, x, generator.block_size, sample=True, top_k=20)
    samples = []
    for gen_mol in y:
        smi = ''.join([itos[int(i)] for i in gen_mol])[1:-1]
        mol = Chem.MolFromSmiles(smi)
        if mol:
            samples.append(smi)
    unique_smiles = list(np.unique(samples))[1:]

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

def get_reward_max(smiles, predictor,adj_fp_fun,x_mutation,x_expr,x_methylation,threshold=1,invalid_reward=0):
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
        if pred_value < threshold:
            #pred_value = -2
            reward = 10


        #reward = np.exp(-pred_value/2)

        return reward

    return invalid_reward


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

train_cell_ic50 =  pd.read_csv('/FasterGTS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])

train_x_mutation = mutation_feature.loc[cell_lines]
train_x_mutation = np.array(train_x_mutation).reshape(train_x_mutation.shape[0],1,train_x_mutation.shape[1],1)

train_x_expr = gexpr_feature.loc[cell_lines]
train_x_expr =  np.array(train_x_expr)

train_x_methylation = methylation_feature.loc[cell_lines]
train_x_methylation = np.array(train_x_methylation)


device = torch.device("cuda:"+str(current_gpu))

from model import ChemGPT

mconf = GPTConfig(n_characters, Max_atoms,
	               n_layer=4, n_head=8, n_embd=256)
generator = ChemGPT(mconf).to(device)

weights = '/FasterGTS/pre_trained_weights/gpt_4layer.pt'
generator.load_state_dict(torch.load(weights))


context = "<"

chars = sorted(list(set(tokens)))

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

n_to_generate = 20
n_policy = 200


config = TrainerConfig(max_epochs=n_policy, batch_size=n_to_generate, learning_rate=5e-5,
	                      lr_decay=True, warmup_tokens=0.1*n_to_generate*n_policy*max_len,
                          final_tokens=n_policy*n_to_generate*max_len,
	                      num_workers=10, ckpt_path = 'cond_gpt/weights/gpt_6layer_rl.pt')

scaler = GradScaler()
optimizer = generator.configure_optimizers(config)

from reinforcement_gpt import Reinforcement
threshold = 0
RL_max = Reinforcement(generator, predictor, get_reward_max,adj_fp_fun, stoi = stoi,
                       x_mutation=train_x_mutation,x_expr=train_x_expr,x_methylation=train_x_methylation)

rewards_max = []
rl_losses_max = []

# Setting up some parameters for the experiment

for i in range(n_policy):
    #for j in range(n_policy):
    generator.eval()
    x = torch.tensor([stoi[context]], dtype=torch.long)[None,...].repeat(n_to_generate, 1).to(device)
    y = get_samples(generator, x, generator.block_size-1, sample=True, top_k=20)

    input_x = []
    samples = []
    for gen_mol in y:
        smi = ''.join([itos[int(i)] for i in gen_mol])[1:-1]
        mol = Chem.MolFromSmiles(smi)
        if mol and len(smi)>3:
            samples.append(smi)
        gen_mol = gen_mol.detach().cpu().numpy()
        del gen_mol

    x = x.detach().cpu().numpy()

    del x, y

    cur_reward, cur_loss = RL_max.policy_gradient(samples, optimizer,scaler,device)
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

    smiles_cur, prediction_cur = estimate_and_update(generator,predictor,
                                                     1000,stoi,itos,
                                                     adj_fp_fun, train_x_mutation,
                                                     train_x_expr,train_x_methylation)

    print('Sample trajectories:')
    smiles_cur,prediction_cur = shuffle(smiles_cur,prediction_cur)
    for sm in smiles_cur[:30]:
        print(sm)
