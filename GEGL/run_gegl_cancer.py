import random
import json
import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from runner.gegl_trainer import GeneticExpertGuidedLearningTrainer
from runner.guacamol_generator import GeneticExpertGuidedLearningGenerator
from model.neural_apprentice import SmilesGenerator, SmilesGeneratorHandler
from model.genetic_expert import GeneticOperatorHandler
from util.storage.priority_queue import MaxRewardPriorityQueue
from util.storage.recorder import Recorder
from util.chemistry.benchmarks import load_benchmark

from util.chemistry.benchmarks import * 
from guacamol.common_scoring_functions import RdkitScoringFunction

from util.smiles.char_dict import SmilesCharDictionary

import neptune.new as neptune

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit.Chem import RDConfig


from data import GeneratorData, MetadataGenerate, GeneratorData, ChembleSmileDataset

import deepchem as dc
from layers.graph import *
import keras

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
device = torch.device(1)

index_list = [1,]
#index_list = [3]
#index_list = [5]
index_list = [7]
index_list = [9]
index_list = [10,11]
#index_list = [13]
#index_list = [15]
index_list = [17]
index_list = [18,19]
index_list = [20]

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

parser = argparse.ArgumentParser(
        description="", formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
parser.add_argument("--benchmark_id", type=int, default=27)
parser.add_argument("--dataset", type=str, default="chembl")
parser.add_argument("--max_smiles_length", type=int, default=100)
parser.add_argument("--apprentice_load_dir", type=str, default="./resource/checkpoint/chembl")
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--sample_batch_size", type=int, default=512)
parser.add_argument("--optimize_batch_size", type=int, default=256)
parser.add_argument("--mutation_rate", type=float, default=0.01)
parser.add_argument("--num_steps", type=int, default=100)
parser.add_argument("--num_keep", type=int, default=1024)
parser.add_argument("--max_sampling_batch_size", type=int, default=256)
parser.add_argument("--apprentice_sampling_batch_size", type=int, default=128)
parser.add_argument("--expert_sampling_batch_size", type=int, default=128)
parser.add_argument("--apprentice_training_batch_size", type=int, default=128)
parser.add_argument("--num_apprentice_training_steps", type=int, default=8)
parser.add_argument("--num_jobs", type=int, default=8)
parser.add_argument("--record_filtered", action="store_true")
args = parser.parse_args()

    
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
    
    # Initialize neptune
    run = neptune.init(
    project="sejin8544/deep-molecular-optimization",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOTJmYWUwNC1lZDhkLTQwZjEtODNhMC03NjI2NmExZDU0NjQifQ==",
    )
 
    run["parameters"] = vars(args)
    run["name"] = "index_"+str(index)
   
   
    
    # Load benchmark, i.e., the scoring function and its corresponding protocol
    
    
    benchmark, scoring_num_list = load_benchmark(args.benchmark_id)
    
    
    benchmark.set_omics(test_x_mutation, test_x_expr, test_x_methylation, target_cell,
                 thres_ic, thres_z, x_mutation, x_expr, x_methylation, predictor, adj_fp_fun, 
                 alpha, beta)
    # Load character directory used for mapping atoms to integers
    char_dict = SmilesCharDictionary(dataset=args.dataset, max_smi_len=args.max_smiles_length)
 
    # Prepare max-reward priority queues
    apprentice_storage = MaxRewardPriorityQueue()
    expert_storage = MaxRewardPriorityQueue()

    # Prepare neural apprentice (we use the weights pretrained on existing dataset)
    apprentice = SmilesGenerator.load(load_dir=args.apprentice_load_dir)
    apprentice = apprentice.to(device)
    apprentice_optimizer = Adam(apprentice.parameters(), lr=args.learning_rate)
    apprentice_handler = SmilesGeneratorHandler(
        model=apprentice,
        optimizer=apprentice_optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.max_sampling_batch_size,
    )
    apprentice.train()

    # Prepare genetic expert
    expert_handler = GeneticOperatorHandler(mutation_rate=args.mutation_rate)

    # Prepare trainer that collect samples from the models & optimize the neural apprentice
    trainer = GeneticExpertGuidedLearningTrainer(
        apprentice_storage=apprentice_storage,
        expert_storage=expert_storage,
        apprentice_handler=apprentice_handler,
        expert_handler=expert_handler,
        char_dict=char_dict,
        num_keep=args.num_keep,
        apprentice_sampling_batch_size=args.apprentice_sampling_batch_size,
        expert_sampling_batch_size=args.expert_sampling_batch_size,
        apprentice_training_batch_size=args.apprentice_training_batch_size,
        num_apprentice_training_steps=args.num_apprentice_training_steps,
        init_smis=[],
        run = run 
    )
    
    trainer.set_omics(test_x_mutation, test_x_expr, test_x_methylation, target_cell,
                 thres_ic, thres_z, x_mutation, x_expr, x_methylation, predictor, adj_fp_fun, 
                 alpha, beta)

    # Prepare recorder that takes care of intermediate logging
    recorder = Recorder(scoring_num_list=scoring_num_list, 
                        record_filtered=args.record_filtered,
                        run = run )

    # Prepare our version of GoalDirectedGenerator for evaluating our algorithm
    guacamol_generator = GeneticExpertGuidedLearningGenerator(
        target_cell = target_cell,
        trainer=trainer,
        recorder=recorder,
        num_steps=args.num_steps,
        device=device,
        scoring_num_list=scoring_num_list,
        num_jobs=args.num_jobs,
    )

    # Run the experiment
    result = benchmark.assess_model(guacamol_generator)
    
    apprentice.save('/NAS_Storage1/leo8544/CanDIS/genetic-expert-guided-learning-main/','chembl_GEGL_'+target_cell)
    
    df = pd.DataFrame(data = trainer.total_sample_list,columns = ['sample','IC50','Z_score','reward'])
    df = df.sort_values(by=['reward'],ascending = False)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
        
    df.to_csv('/NAS_Storage1/leo8544/CanDIS/genetic-expert-guided-learning-main/results/chembl_GEGL_'+target_cell+'_final.csv')
    
    
    #scoring_function = benchmark.wrapped_objective
    #pool = guacamol_generator.pool
    #expert_smis, expert_scores = guacamol_generator.trainer.update_storage_by_expert(scoring_function, guacamol_generator.pool)
    
    
    # Dump the final result to neptune
    run['benchmark_score'] = result.score
    #neptune.set_property("benchmark_score", result.score)
