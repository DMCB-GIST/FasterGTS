#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 04:14:17 2021

@author: akshat
"""
index_list = [0]
#index_list = [1]
#index_list = [2]

#index_list = [3]
#index_list = [4]
#index_list = [5]
#index_list = [6,7]
#index_list = [8,9]
#index_list = [10,11]
#index_list = [12,13]
#index_list = [14,15]
#index_list = [16,17]
#index_list = [18,19]
#index_list = [20]

import os
import numpy as np 
import random
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from SAS_calculator.sascorer import calculateScore
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
import pandas as pd
import dask.dataframe as dd
import multiprocessing

from mutate import get_mutated_smiles
from crossover import crossover_smiles

import selfies
from selfies import encoder, decoder

# Updated SELFIES constraints: 
default_constraints = selfies.get_semantic_constraints()
new_constraints = default_constraints
new_constraints['S'] = 2
new_constraints['P'] = 3
selfies.set_semantic_constraints(new_constraints)  # update constraints


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from data import GeneratorData, MetadataGenerate, GeneratorData, ChembleSmileDataset

import deepchem as dc
from layers.graph import *
import keras

# from tensorboardX import SummaryWriter
# writer = SummaryWriter()

def sanitize_smiles(smi):
    '''Return a canonical smile representation of smi
    
    Parameters:
    smi (string) : smile string to be canonicalized 
    
    Returns:
    mol (rdkit.Chem.rdchem.Mol) : RdKit mol object                          (None if invalid smile string smi)
    smi_canon (string)          : Canonicalized smile representation of smi (None if invalid smile string smi)
    conversion_successful (bool): True/False to indicate if conversion was  successful 
    '''
    try:
        mol = smi2mol(smi, sanitize=True)
        smi_canon = mol2smi(mol, isomericSmiles=False, canonical=True)
        return (mol, smi_canon, True)
    except:
        return (None, None, False)

def get_fp_scores(smiles_back, target_smi): 
    smiles_back_scores = []
    target    = Chem.MolFromSmiles(target_smi)
    fp_target = AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back: 
        mol    = Chem.MolFromSmiles(item)
        fp_mol = AllChem.GetMorganFingerprint(mol, 2)
        score  = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores

def reward_fun(IC50,z_score):
    if z_score<=thres_z and IC50<= thres_ic:
        return np.exp(alpha*(-z_score+thres_z))+beta*np.log(-IC50+1+thres_ic)
    else:
        return np.exp(alpha*(-z_score+thres_z))

def PlP(smiles):
    reward = -10000
    try:
        mol = Chem.MolFromSmiles(smiles)
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
    
        reward = reward_fun(IC50,z_score)
        
    except:
        print(smiles)
        pass
    
    return reward

def PlP1(smiles):
    mol   = Chem.MolFromSmiles(smiles)
    log_P = (Descriptors.MolLogP(mol) - 2.4729421499641497) / 1.4157879815362406
    sas_  = (calculateScore(mol) - 3.0470797085649894) / 0.830643172314514
    
    cycle_list = mol.GetRingInfo().AtomRings() 
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6

    cycle_length = (cycle_length - 0.038131530820234766) / 0.2240274735210179
    return log_P-sas_-cycle_length


def parr_property(df_in):
    return df_in.SMILES.apply(PlP)

def get_random_smiles(num_random): 

    alphabet = ['[=N]', '[C]', '[S]','[Branch3_1]','[Expl=Ring3]','[Branch1_1]','[Branch2_2]','[Ring1]', '[#P]','[O]', '[Branch2_1]', '[N]','[=O]','[P]','[Expl=Ring1]','[Branch3_2]','[I]', '[Expl=Ring2]', '[=P]','[Branch1_3]','[#C]','[Cl]', '[=C]','[=S]','[Branch1_2]','[#N]','[Branch2_3]','[Br]','[Branch3_3]','[Ring3]','[Ring2]','[F]'] +  ['[C][=C][C][=C][C][=C][Ring1][Branch1_2]']*2 
    max_smi_len = 100
    collect_random = []
    
    for _ in range(num_random): 
        random_len = random.randint(1, max_smi_len+1)
        random_alphabets = list(np.random.choice(alphabet, random_len)) 
        random_selfies = ''.join(x for x in random_alphabets)
        
        collect_random.append(decoder(random_selfies))
    
    return [x for x in collect_random if x != '']

def get_good_bad_smiles(fitness, population, generation_size): 
    
    fitness             = np.array(fitness)
    idx_sort            = fitness.argsort()[::-1] # Best -> Worst
    
    keep_ratio          = 0.2
    keep_idx            = int(len(list(idx_sort)) * keep_ratio)
    F_50_val  = fitness[idx_sort[keep_idx]]

    F_25_val = (np.array(fitness) - F_50_val)
    F_25_val = np.array([x for x in F_25_val if x<0]) + F_50_val
    F_25_sort = F_25_val.argsort()[::-1]
    F_25_val = F_25_val[F_25_sort[0]]

    prob_   = 1 / ( 3**((F_50_val-fitness) / (F_50_val-F_25_val)) + 1 )
    
    prob_   = prob_ / sum(prob_)  
    to_keep = np.random.choice(generation_size, keep_idx, p=prob_)    
    to_replace = [i for i in range(generation_size) if i not in to_keep][0: generation_size-len(to_keep)]
    
    keep_smiles = [population[i] for i in to_keep]    
    replace_smiles = [population[i] for i in to_replace]
    
    best_smi = population[idx_sort[0]]
    if best_smi not in keep_smiles: 
        keep_smiles.append(best_smi)
        if best_smi in replace_smiles: replace_smiles.remove(best_smi)

    return keep_smiles, replace_smiles



Drug_info_file = '/NAS_Storage1/leo8544/CanDIS/data/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'
Cell_line_info_file = '/NAS_Storage1/leo8544/CanDIS/data/CCLE/Cell_lines_annotations_20181226.txt'
Drug_feature_file = '/NAS_Storage1/leo8544/CanDIS/data/GDSC/drug_graph_feat'
Genomic_mutation_file = '/NAS_Storage1/leo8544/CanDIS/data/CCLE/genomic_mutation_34673_demap_features.csv'
Gene_expression_file = '/NAS_Storage1/leo8544/CanDIS/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = '/NAS_Storage1/leo8544/CanDIS/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'
    
    
checkpoint = '/NAS_Storage1/leo8544/CanDIS/pre_trained_weights/adj_best_DeepCDR_with_mut_with_gexp_with_methy_256_256_256_bn_relu_GAP.h5'
predictor = keras.models.load_model(checkpoint,custom_objects={'GraphConv':GraphConv})

try:
    from deepchem.feat.adjacency_fingerprint import AdjacencyFingerprint
    adj_fp_fun  = AdjacencyFingerprint(max_n_atoms=100)
except:
    adj_fp_fun  = dc.feat.adjacency_fingerprint.AdjacencyFingerprint(max_n_atoms=100)
    
mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,
                                                                                                    Cell_line_info_file,
                                                                                                    Genomic_mutation_file,
                                                                                                    Drug_feature_file,
                                                                                                    Gene_expression_file,
                                                                                                    Methylation_file,False)
    
    
train_cell_ic50 =  pd.read_csv('/NAS_Storage1/leo8544/CanDIS/train_cell_ic50.csv',sep=',')
cell_lines = list(train_cell_ic50['cell line'])
    
TNBC_cells = pd.read_csv("/NAS_Storage1/leo8544/CanDIS/TNBC_cells_info.csv",index_col = 0)
        
test_x_mutation = mutation_feature.loc[cell_lines]
test_x_mutation = np.array(test_x_mutation).reshape(test_x_mutation.shape[0],1,test_x_mutation.shape[1],1)
        
test_x_expr = gexpr_feature.loc[cell_lines]
test_x_expr =  np.array(test_x_expr)  
        
test_x_methylation = methylation_feature.loc[cell_lines]
test_x_methylation = np.array(test_x_methylation)


alpha = 0.8
beta = 1.8  


for index in index_list:
    target_cell = TNBC_cells.index[index]
    thres_ic = TNBC_cells['Thres_IC50'][index]
    thres_z = TNBC_cells['Thres_Z'][index]
    
    x_mutation = mutation_feature.loc[target_cell]
    x_mutation = np.array(x_mutation).reshape(1,1,x_mutation.shape[0],1)
        
    x_expr = gexpr_feature.loc[target_cell]
    x_expr = np.array(x_expr).reshape(1,x_expr.shape[0])
        
    x_methylation = methylation_feature.loc[target_cell]
    x_methylation = np.array(x_methylation).reshape(1,x_methylation.shape[0])
        
    
    initial_mol      = ['C'] + get_random_smiles(20) # Add on 20 random smiles
    smiles_collector = {} # A tracker for all smiles! 
    generations      = 50
    generation_size  = 250
    num_mutation_ls  = [5]
    mutn_prob        = 0.75       # chance of mutation; else a crossover is performed 
    choice_ls        = [1, 2, 3] # Insert; replace; delete 
    total_val_num = 0
            
    population = np.random.choice(initial_mol, size=generation_size).tolist()
    
    
    # Calculate fitness for the initial population: 
    unique_pop     = list(set(population))
    df             = pd.DataFrame(unique_pop, columns=['SMILES'])
    num_cores      = multiprocessing.cpu_count() 
    #ddf            = dd.from_pandas(df, npartitions=num_cores)
    #scheduler='processes'
    df['property'] = [PlP(i) for i in df['SMILES']]
    #ddf.map_partitions(parr_property, meta='float').compute()
    prop_map       = df.set_index('SMILES').to_dict()['property']
    fitness        = []
    for item in population: 
        fitness.append(prop_map[item])
        
    # Save fitness onto global collector: 
    for item in prop_map: 
        smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]
    
    total_val_num += len(smiles_collector)
    for gen_ in range(generations): 
        
        
        # STEP 1: OBTAIN THE NEXT GENERATION OF MOLECULES (FOR EXPLORATION): 
        # Order based on fitness, and decide which part of the population is to be kept/discarded: 
        keep_smiles, replace_smiles = get_good_bad_smiles(fitness, population, generation_size)
        replace_smiles = list(set(replace_smiles))

        # Mutations:     
        mut_smi_dict = get_mutated_smiles(replace_smiles[0: len(replace_smiles)//2]) # Half the molecuules are to be mutated     
        # Crossovers: 
        smiles_join = []
        for item in replace_smiles[len(replace_smiles)//2: ]: 
            smiles_join.append(item + 'xxx' + random.choice(keep_smiles))
        cross_smi_dict =  crossover_smiles(smiles_join)                              # Other half of the molecules are to be crossed over
        
        all_mut_smiles = []
        for key in mut_smi_dict: 
            all_mut_smiles.extend(mut_smi_dict[key])
        all_mut_smiles = list(set(all_mut_smiles))
        all_mut_smiles = [x for x in all_mut_smiles if x != '']
        
        all_cros_smiles = []
        for key in cross_smi_dict: 
            all_cros_smiles.extend(cross_smi_dict[key])
        all_cros_smiles = list(set(all_cros_smiles))
        all_cros_smiles = [x for x in all_cros_smiles if x != '']
        
        all_smiles = list(set(all_mut_smiles + all_cros_smiles))
        all_smiles_unique = [x for x in all_smiles if x not in smiles_collector]
        
        total_val_num += len(all_smiles_unique)
        
        # STEP 2: CONDUCT FITNESS CALCULATION FOR THE EXPLORATION MOLECULES: 
        replaced_pop = random.sample(all_smiles_unique, generation_size-len(keep_smiles))
        population   = keep_smiles + replaced_pop
        
        unique_pop     = list(set(population))
        df             = pd.DataFrame(unique_pop, columns=['SMILES'])
        #num_cores      = multiprocessing.cpu_count() 
        #ddf            = dd.from_pandas(df, npartitions=num_cores)
        df['property'] = [PlP(i) for i in df['SMILES']]#ddf.map_partitions(parr_property, meta='float').compute(scheduler='processes')
        prop_map       = df.set_index('SMILES').to_dict()['property']
        fitness        = []
        for item in population: 
            if item in prop_map: 
                fitness.append(prop_map[item])
            else: 
                fitness.append(smiles_collector[item][0])
            
        # Save fitness onto global collector: 
        for item in population: 
            if item not in smiles_collector: 
                smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]
            else: 
                smiles_collector[item] = [smiles_collector[item][0], smiles_collector[item][1]+1]
        
        print('On generation {}/{}'.format(gen_, generations) )
        idx_sort = np.argsort(fitness)[::-1]
        top_idx = idx_sort[0]
        print('    (Explr) Top Fitness: {}'.format(fitness[top_idx]))
        print('    (Explr) Top Smile: {}'.format(population[top_idx]))
        
        
        # Logging: 
        results_dir_path = './RESULTS/'
        #if not os.path.isdir(results_dir_path): 
        #    os.system('mkdir {}'.format(results_dir_path))            
        fitness_sort = [fitness[x] for x in idx_sort]
        #with open(results_dir_path + '/fitness_explore_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['{}\n'.format(x) for x in fitness_sort])
        population_sort = [population[x] for x in idx_sort]
        #with open(results_dir_path + '/population_explore_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['{}\n'.format(x) for x in population_sort])
                    
        
        # TODO: For the NN! 
        population_remaining = [x for x in all_smiles_unique if x not in smiles_collector]
        
        # STEP 3: CONDUCT LOCAL SEARCH: 
        # top_mols = 1
        # top_idx = np.argsort(fitness)[::-1][0: top_mols]
        smiles_local_search = [population[top_idx]]
        mut_smi_dict_local  = get_mutated_smiles(smiles_local_search)
        mut_smi_dict_local  = mut_smi_dict_local[population[top_idx]]
        mut_smi_dict_local  = [x for x in mut_smi_dict_local if x not in smiles_collector]

        fp_scores          = get_fp_scores(mut_smi_dict_local, population[top_idx])
        fp_sort_idx        = np.argsort(fp_scores)[::-1][: generation_size]
        mut_smi_dict_local_calc = [mut_smi_dict_local[i] for i in fp_sort_idx]
        
        total_val_num += len(mut_smi_dict_local)
        
        # STEP 4: CALCULATE THE FITNESS FOR THE LOCAL SEARCH: 
        df             = pd.DataFrame(mut_smi_dict_local_calc, columns=['SMILES'])
        #ddf            = dd.from_pandas(df, npartitions=num_cores)
        df['property'] = [PlP(i) for i in df['SMILES']]#ddf.map_partitions(parr_property, meta='float').compute(scheduler='processes')
        prop_map       = df.set_index('SMILES').to_dict()['property']
        fitness_local_search = []
        for item in mut_smi_dict_local_calc: 
            if item in prop_map: 
                fitness_local_search.append(prop_map[item])
            else: 
                fitness.append(smiles_collector[item][0])
        
        idx_sort_lc = np.argsort(fitness_local_search)[::-1]
        print('    (Local) Top Fitness: {}'.format(fitness_local_search[idx_sort_lc[0]]))
        print('    (local) Top Smile: {}'.format(mut_smi_dict_local_calc[idx_sort_lc[0]]))
        
        # Store the results: 
        for item in mut_smi_dict_local_calc: 
            if item not in smiles_collector: 
                smiles_collector[item] = [prop_map[item], 1] # [Property_value, Count]
            else: 
                smiles_collector[item] = [smiles_collector[item][0], smiles_collector[item][1]+1]
        
        # TODO: For the NN! 
        mut_smi_dict_local_remain = [x for x in mut_smi_dict_local if x not in mut_smi_dict_local_calc]
        
        
        
        # Logging: 
        fitness_sort = [fitness_local_search[x] for x in idx_sort_lc]
        #with open(results_dir_path + '/fitness_local_search_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['{}\n'.format(x) for x in fitness_sort])
        population_sort = [mut_smi_dict_local_calc[x] for x in idx_sort_lc]
        #with open(results_dir_path + '/population_local_search_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['{}\n'.format(x) for x in population_sort])
        
        # TODO: Uncomment
        #sns.distplot(fitness_local_search, label='Local Search')
        #sns.distplot(fitness, label='Exploration')
        #plt.legend()
        #plt.savefig(results_dir_path+'/prop_distributions.png', dpi=1000)
        #plt.show()
                
        
        
        # STEP 5: EXCHANGE THE POPULATIONS: 
        
        # Introduce changes to 'fitness' & 'population'
        # With replacesments from 'fitness_local_search' & 'mut_smi_dict_local_calc'
        num_exchanges     = 5
        introduce_smiles  = population_sort[0:num_exchanges] # Taking the top 5 molecules
        introduce_fitness = fitness_sort[0:num_exchanges]    # Taking the top 5 molecules
        
        worst_indices = idx_sort[-num_exchanges: ]
        for i,idx in enumerate(worst_indices): 
            population[idx] = introduce_smiles[i]
            fitness[idx]    = introduce_fitness[i]
        
        #with open(results_dir_path + '/exchange_members_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['{}, {}\n'.format(introduce_smiles[i], introduce_fitness[i]) for i in range(len(introduce_fitness))])
        
        # Save best of generation!: 
        fit_all_best = np.argmax(fitness)
        #with open('./RESULTS' + '/generation_all_best_'+str(gen_)+'th_'+target_cell+'.txt', 'a+') as f: 
        #    f.writelines(['Gen:{}, {}, {}\n'.format(gen_,  population[fit_all_best], fitness[fit_all_best])])
        
        keys = smiles_collector.keys()
        values = list(smiles_collector.values())
        reward = [i[0] for i in values]
        df = pd.DataFrame(index = keys, data = reward, columns = ['reward']).sort_values(by=['reward'],ascending=False)
        df.to_csv('./RESULTS/'+target_cell+'_'+str(gen_)+'_th_results.csv')
                    
        print('############### smiles_collector: ', len(smiles_collector))
