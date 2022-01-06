import pandas as pd
import argparse
from utils import set_seed
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model import GPTConfig, ChemGPT
from trainer import ChembleTrainer, TrainerConfig
from data import ChembleSmileDataset
import math

from sklearn.model_selection import train_test_split

from data import GeneratorData

gen_data_path = '/FasterGTS/data/ChEMBL/chembl_22_clean_1576904_sorted_std_final.smi'

tokens = ['<', '>', '#', '%', ')', '(', '+', '-', '/', '.', '1', '0', '3', '2', '5', '4', '7',
              '6', '9', '8', '=', 'A', '@', 'C', 'B', 'F', 'I', 'H', 'O', 'N', 'P', 'S', '[', ']',
              '\\', 'c', 'e', 'i', 'l', 'o', 'n', 'p', 's', 'r', '\n', '$']

Max_atoms = 101
n_characters = 46

current_gpu = 0
try:
    print("already data ",gen_data)
except:
    gen_data = GeneratorData(training_data_path=gen_data_path, delimiter='\t',
                           cols_to_read=[0], keep_header=True, tokens=tokens)


    smiles =[ i for i in gen_data.file if len(i) < Max_atoms ]
    smiles, vsmiles = train_test_split(smiles, test_size=0.01, random_state=42)
             #[ i[1:-1] for i in gen_data.file]
    #lens = [len(regex.findall(i.strip())) for i in smiles]

    max_len = Max_atoms


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False, help='debug')
    parser.add_argument('--n_layer', type=int, default = 2, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default = 5, help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default = 512, help="batch size", required=False)
    parser.add_argument('--learning_rate', type=int, default = 6e-4, help="learning rate", required=False)

    args = parser.parse_args()
    set_seed(42)
    smiles = [ i + str('$')*(max_len - len(i)) for i in smiles]
    vsmiles = [ i + str('$')*(max_len - len(i)) for i in vsmiles]


    train_dataset = ChembleSmileDataset(smiles,tokens,max_len)
    valid_dataset = ChembleSmileDataset(vsmiles,tokens,max_len)

    from model import ChemGPT
    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)

    model = ChemGPT(mconf)#.to(device)

    config = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
	                      lr_decay=True, warmup_tokens=0.1*len(smiles)*max_len,
                          final_tokens=args.max_epochs*len(smiles)*max_len,
	                      num_workers=10, ckpt_path = '/FasterGTS/pre_trained_weights/gpt_2layer.pt')

    from trainer import ChembleTrainer

    trainer = ChembleTrainer(model, train_dataset, valid_dataset, config,current_gpu=current_gpu)

    trainer.train()
