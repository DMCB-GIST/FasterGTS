"""
This code is written by authors of ExPerT.

All rights reserved.
"""

from anytree import Node,AnyNode,RenderTree

import torch
import torch.nn as nn
import time
import math
import random
import pandas as pd

from anytree.exporter import DotExporter
from anytree.importer import DictImporter
from anytree.exporter import DictExporter
import numpy as np
import matplotlib.pyplot as plt

from rdkit import Chem
import scaffoldgraph as sg

from collections import Counter

import operator

from queue import PriorityQueue

import heapq

from GA_functions import reproduce

from sklearn.utils import shuffle

from data import ChembleSmileDataset

NO_REWARD = -10000

class State_trainable_RNNs():
    def __init__(self,generator, trainable_generator, predictor,adj_fp_fun,
                 mutation_feature,gexpr_feature,methylation_feature, cell_lines,target_cell,
                 thres_ic = 0,thres_z = -1,numSample=10, alpha = 1, beta = 2,
                 train_batch=256, max_queue=1024, data=None,generation_ratio = 0.8,
                 num_self = 30, mutation_rate = 0.01, num_GA =30):
        self.generator = generator
        self.trainable_generator = trainable_generator
        self.predictor = predictor
        self.numSample = numSample
        self.data = data
        self.adj_fp_fun = adj_fp_fun

        self.end_token = '>'

        self.total_sample_list = []

        self.best_worst_score = -1000

        self.best_samples = []
        self.best_scores = []

        self.total_reward = 0
        self.winning_count = 0
        self.valid_count = 1

        self.thres_ic = thres_ic
        self.thres_z = thres_z
        self.target_cell = target_cell

        self.cell_lines = cell_lines

        self.x_mutation = mutation_feature.loc[self.target_cell]
        self.x_mutation = np.array(self.x_mutation).reshape(1,1,self.x_mutation.shape[0],1)


        self.x_expr = gexpr_feature.loc[self.target_cell]
        self.x_expr = np.array(self.x_expr).reshape(1,self.x_expr.shape[0])

        self.x_methylation = methylation_feature.loc[self.target_cell]
        self.x_methylation = np.array(self.x_methylation).reshape(1,self.x_methylation.shape[0])

        self.test_x_mutation = mutation_feature.loc[self.cell_lines]
        self.test_x_mutation = np.array(self.test_x_mutation).reshape(self.test_x_mutation.shape[0],1,self.test_x_mutation.shape[1],1)

        self.test_x_expr = gexpr_feature.loc[self.cell_lines]
        self.test_x_expr =  np.array(self.test_x_expr)

        self.test_x_methylation = methylation_feature.loc[self.cell_lines]
        self.test_x_methylation = np.array(self.test_x_methylation)

        self.train_batch = train_batch
        self.que = PriorityQueue() #maxsize=max_queue
        self.max_queue = max_queue

        self.alpha = alpha
        self.beta = beta

        self.generation_ratio = generation_ratio
        self.num_self = num_self

        self.num_GA = num_GA

        self.mutation_rate = mutation_rate

    def getUniqueList(self):
        return list(set(self.total_sample_list))

    def getPossibleActions(self,current_molecule,numSimul=10):
        action_list = self.generator.get_next_actions(self.data, current_molecule=current_molecule,numActions=numSimul)
        return action_list

    def getSampleList(self,current_molecule='<'):

        if current_molecule[-1] == self.end_token:
            return [current_molecule]

        sample_list = self.getSamples(current_molecule,self.numSample)

        return sample_list

    def getSamples(self,current_molecule,num_samples):
        samples = []
        with torch.no_grad():
            self.trainable_generator.eval()
            samples += self.trainable_generator.new_simulation(self.data, current_molecule=current_molecule
                                                               ,numSimul = int(num_samples*self.generation_ratio))
            samples += self.generator.new_simulation(self.data, current_molecule=current_molecule,
                                                     numSimul = int(num_samples*(1-self.generation_ratio)))

        return samples

    def reward(self,IC50,z_score):
        if z_score<=self.thres_z and IC50<= self.thres_ic:
            return np.exp(self.alpha*(-z_score+self.thres_z))+self.beta*np.log(-IC50+1+self.thres_ic)
        else:
            return 1

    def get_best_smis(self):
        return list(zip(*self.que.queue))[1]

    def get_best_scores(self):
        return list(zip(*self.que.queue))[0]

    def get_input(self,smi):
        inp = self.data.char_tensor(smi[:-1])
        target =self.data.char_tensor(smi[1:])
        return inp, target


    def check_valid_smi(self,smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return smi
        else:
            return None

    def get_smis(self,num_smis):
        smis = []
        while len(smis) != num_smis:
            smi = self.generator.new_simulation(self.data, current_molecule='<',
                                                     numSimul = 1)[0]
            if self.check_valid_smi(smi[1:-1]):
                smis.append(smi)

        return smis


    def GA(self):
        smis = random.choices(self.get_best_smis(), k= self.num_GA)
        smis += self.get_smis(self.num_GA)
        smis = shuffle(smis)

        smi0s, smi1s = smis[:self.num_GA], smis[self.num_GA:]

        new_smis = [reproduce(smi0s[i][1:-1],smi1s[i][1:-1],self.mutation_rate) for i in range(self.num_GA)]

        smis = []
        rewards = []
        for smi in new_smis:
            IC50, z_score, reward = self.cal_reward(smi)

            if reward > 1:
                update = self.update_que(reward, '<'+smi+'>')
                #self.total_sample_list.append('<'+smi+'>'+str(round(IC50,4))+'>'+str(z_score))

                if update:
                    smis.append(smi)
                    rewards.append(reward)

        return smis, rewards


    def self_train(self):
        smis = []
        rewards = []
        samples = self.getSamples('<',self.num_self)
        for smi in samples:
            IC50,z_score,reward = self.cal_reward(smi[1:-1])
            #self.total_sample_list.append(smi+str(round(IC50,4))+'>'+str(z_score))

            if reward > 1:
                update = self.update_que(reward, smi)

                if update:
                    smis.append(smi)
                    rewards.append(reward)

        return smis, rewards

    def get_loss(self,inp, target):
        hidden = self.trainable_generator.init_hidden()
        if self.trainable_generator.has_cell:
            cell = self.trainable_generator.init_cell()
            hidden = (hidden, cell)
        if self.trainable_generator.has_stack:
            stack = self.trainable_generator.init_stack()
        else:
            stack = None

        loss = 0.0
        for c in range(len(inp)):
            output, hidden, stack = self.trainable_generator(inp[c], hidden, stack)
            loss += self.trainable_generator.criterion(output, target[c].unsqueeze(0))

        return loss/len(inp)

    def step(self):

        avg_loss = 0.0
        smis = self.get_best_smis()
        batch = min(self.train_batch,len(self.que.queue)) #int(len(self.que.queue)/2))
        rand_smis = random.choices(population=smis, k=batch)

        self.trainable_generator.train()
        #self.trainable_generator.optimizer.zero_grad()

        for smi in rand_smis:
            inp, target = self.get_input(smi)
            #avg_loss += self.get_loss(inp, target)
            avg_loss += self.trainable_generator.train_step(inp, target)

        avg_loss /= batch

        #avg_loss.backward()
        #self.trainable_generator.optimizer.step()

        #avg_loss = avg_loss.item()
        #print('avg_loss ',avg_loss)

        return avg_loss

    def update_que(self,reward, sample):
        try:
            self.que.queue.index((reward,sample))
            return False
        except:
            if self.max_queue <= len(self.que.queue) and reward > self.best_worst_score:
                self.que.get()

            if self.max_queue > len(self.que.queue):
                self.que.put((reward, sample))
                worst = self.que.get()
                self.best_worst_score = worst[0]
                self.que.put(worst)

                return True

    def cal_reward(self,smi):
        IC50 = 0
        z_score = 0
        reward = NO_REWARD
        try:
            mol = Chem.MolFromSmiles(smi)
            adj_fp_data =  self.adj_fp_fun.featurize([mol])
            single_adj_data = adj_fp_data[0][0].reshape(1,100,100)
            single_drug_feat_data = adj_fp_data[0][1].reshape(1,100,75)

            adj_data = np.tile(single_adj_data,[self.x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[self.x_mutation.shape[0],1,1])
            input_data = [drug_feat_data,adj_data, self.x_mutation,self.x_expr,self.x_methylation]

            IC50 = self.predictor.predict(input_data)

            IC50 = np.average(list(IC50))

            adj_data = np.tile(single_adj_data,[self.test_x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[self.test_x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, self.test_x_mutation,self.test_x_expr,self.test_x_methylation]
            test_pred_value = self.predictor.predict(input_data)
            test_pred_value = test_pred_value.reshape(test_pred_value.shape[0])
            mean = np.mean(test_pred_value)
            std = np.std(test_pred_value)

            z_score = round((IC50-mean)/std,2)

            reward = self.reward(IC50,z_score)

            self.valid_count += 1
            self.total_reward += reward

            self.total_sample_list.append('<'+smi+'>'+str(round(IC50,4))+'>'+str(z_score))


            if reward > 1:
                self.winning_count += 1

        except:
            pass


        return IC50, z_score, reward

    def getReward(self,sample_list):
        total_pred_reward = 0
        valid_count = 0
        winning_count = 0
        bestReward = NO_REWARD
        bestSample = ""
        samples = []

        for sample in sample_list:

            IC50, z_score, reward = self.cal_reward(sample[1:-1])

            if reward > NO_REWARD:
                total_pred_reward += reward
                valid_count +=1
                #self.total_sample_list.append(sample+str(round(IC50,4))+'>'+str(z_score))

                if reward > 1:
                    samples.append(sample[1:-1])
                    winning_count += 1

                    self.update_que(reward, sample)


                if reward > bestReward:
                    bestReward = reward
                    bestSample = sample

        #self.valid_count += valid_count
        #self.total_reward += total_pred_reward
        #self.winning_count += winning_count

        return total_pred_reward, valid_count, bestReward, bestSample, winning_count, samples



class State_trainable_GPT():
    def __init__(self,generator, trainable_generator, predictor,adj_fp_fun,stoi,itos,
                 mutation_feature,gexpr_feature,methylation_feature,
                 cell_lines,target_cell,thres_ic = 0,thres_z = -1,numSample=10, alpha = 1, beta = 2,
                 train_batch=256, max_queue=1024,generation_ratio = 0.8,
                 num_self = 30, mutation_rate = 0.01, num_GA =30,
                 optimizer = None, scaler = None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.generator = generator
        self.trainable_generator = trainable_generator
        self.predictor = predictor
        self.numSample = numSample
        self.adj_fp_fun = adj_fp_fun

        self.stoi = stoi
        self.itos = itos
        self.data = stoi

        self.end_token = '>'

        self.total_sample_list = []

        self.max_len = 100
        self.total_reward = 0
        self.winning_count = 0
        self.valid_count = 1
        self.thres_ic = thres_ic
        self.thres_z = thres_z
        self.target_cell = target_cell

        self.cell_lines = cell_lines

        self.x_mutation = mutation_feature.loc[self.target_cell]
        self.x_mutation = np.array(self.x_mutation).reshape(1,1,self.x_mutation.shape[0],1)


        self.x_expr = gexpr_feature.loc[self.target_cell]
        self.x_expr = np.array(self.x_expr).reshape(1,self.x_expr.shape[0])

        self.x_methylation = methylation_feature.loc[self.target_cell]
        self.x_methylation = np.array(self.x_methylation).reshape(1,self.x_methylation.shape[0])

        self.test_x_mutation = mutation_feature.loc[self.cell_lines]
        self.test_x_mutation = np.array(self.test_x_mutation).reshape(self.test_x_mutation.shape[0],1,self.test_x_mutation.shape[1],1)

        self.test_x_expr = gexpr_feature.loc[self.cell_lines]
        self.test_x_expr =  np.array(self.test_x_expr)

        self.test_x_methylation = methylation_feature.loc[self.cell_lines]
        self.test_x_methylation = np.array(self.test_x_methylation)

        self.alpha = alpha
        self.beta = beta

        self.train_batch = train_batch
        self.que = PriorityQueue() #maxsize=max_queue
        self.max_queue = max_queue


        self.generation_ratio = generation_ratio
        self.num_self = num_self

        self.num_GA = num_GA

        self.mutation_rate = mutation_rate

        self.token = list(itos.values())

        self.device = self.trainable_generator.state_dict()['head.weight'].device


    def getUniqueList(self):
        return list(set(self.total_sample_list))

    def getPossibleActions(self,current_molecule='<',numSimul=10):
        action_list = self.generator.get_next_actions(self.stoi,self.itos,numSimul,current_molecule=current_molecule)
        return action_list

    def getSampleList(self,current_molecule='<'):

        if current_molecule[-1] == self.end_token:
            return [current_molecule]

        sample_list = self.getSamples(current_molecule,self.numSample)

        return sample_list

    def getSamples(self,current_molecule,num_samples):
       samples = []
       with torch.no_grad():
           self.trainable_generator.eval()
           samples += self.trainable_generator.get_samples(self.stoi,self.itos,int(num_samples*self.generation_ratio),
                                                  current_molecule=current_molecule)
           samples += self.generator.get_samples(self.stoi,self.itos,int(num_samples*(1-self.generation_ratio)),
                                                  current_molecule=current_molecule)

       return samples

    def reward(self,IC50,z_score):
        if z_score<=self.thres_z and IC50<= self.thres_ic:
            return np.exp(self.alpha*(-z_score+self.thres_z))+self.beta*np.log(-IC50+1+self.thres_ic)
        else:
            return 1

    def get_best_smis(self):
        return list(zip(*self.que.queue))[1]

    def get_best_scores(self):
        return list(zip(*self.que.queue))[0]

    def check_valid_smi(self,smi):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return smi
        else:
            return None

    def get_smis(self,num_smis):
        smis = []
        while len(smis) != num_smis:
            try:
                smi = self.generator.get_samples(self.stoi,self.itos, n_samples = 1, current_molecule='<')
                smi = smi[0]
                if self.check_valid_smi(smi[1:-1]):
                    smis.append(smi)
            except:
                print('error')
                print(smi)

        return smis

    def GA(self,save_all = False):
        smis = random.choices(self.get_best_smis(), k= self.num_GA)
        smis += self.get_smis(self.num_GA)
        smis = shuffle(smis)

        smi0s, smi1s = smis[:self.num_GA], smis[self.num_GA:]

        new_smis = [reproduce(smi0s[i][1:-1],smi1s[i][1:-1],self.mutation_rate) for i in range(self.num_GA)]

        smis = []
        rewards = []
        for smi in new_smis:
            IC50, z_score, reward = self.cal_reward(smi)

            if reward > 0 and save_all:
                smis.append(smi)
                rewards.append(reward)

            if reward > 1:
                update = self.update_que(reward, '<'+smi+'>')

                if update and not(save_all):
                    smis.append(smi)
                    rewards.append(reward)

        return smis, rewards

    def get_input(self,smi):
        tensor_2 = torch.tensor([2])

        inp = torch.tensor([self.stoi[i] for i in smi])
        inp = torch.cat((inp,tensor_2.repeat(self.max_len-len(inp))))
        target = torch.tensor([self.stoi[i] for i in smi[1:]])
        target = torch.cat((target,tensor_2.repeat(self.max_len-len(target))))

        return inp, target

    def step(self):

        self.trainable_generator.train()

        losses = []
        smis = self.get_best_smis()
        batch = min(self.train_batch,len(self.que.queue)) #int(len(self.que.queue)/2))
        rand_smis = random.choices(population=smis, k=batch)

        xs = []
        ys = []
        for smi in rand_smis:
            x,y = self.get_input(smi)
            xs.append(x)
            ys.append(y)

        x = torch.cat(xs).reshape(len(rand_smis),self.max_len).to(self.device)
        y = torch.cat(ys).reshape(len(rand_smis),self.max_len).to(self.device)

        with torch.cuda.amp.autocast():
            logits, loss = self.trainable_generator(x, y)
            loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
            losses.append(loss.item())

        self.trainable_generator.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.trainable_generator.parameters(), 1)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return float(np.mean(losses))


    def self_train(self,save_all = False):
        smis = []
        rewards = []
        samples = self.getSamples('<',self.num_self)
        for smi in samples:
            IC50,z_score,reward = self.cal_reward(smi[1:-1])

            if reward >0 and save_all:
                smis.append(smi[1:-1])
                rewards.append(reward)

            if reward > 1:
                update = self.update_que(reward, smi)

                if update and not(save_all):
                    smis.append(smi)
                    rewards.append(reward)


        return smis, rewards

    def update_que(self,reward, sample):
        try:
            self.que.queue.index((reward,sample))
            return False
        except:
            if self.max_queue <= len(self.que.queue) and reward > self.best_worst_score:
                self.que.get()

            if self.max_queue > len(self.que.queue):
                self.que.put((reward, sample))
                worst = self.que.get()
                self.best_worst_score = worst[0]
                self.que.put(worst)

                return True

    def cal_reward(self,smi):
        IC50 = 0
        z_score = 0
        reward = NO_REWARD
        try:
            mol = Chem.MolFromSmiles(smi)
            adj_fp_data =  self.adj_fp_fun.featurize([mol])
            single_adj_data = adj_fp_data[0][0].reshape(1,100,100)
            single_drug_feat_data = adj_fp_data[0][1].reshape(1,100,75)

            adj_data = np.tile(single_adj_data,[self.x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[self.x_mutation.shape[0],1,1])
            input_data = [drug_feat_data,adj_data, self.x_mutation,self.x_expr,self.x_methylation]

            IC50 = self.predictor.predict(input_data)

            IC50 = np.average(list(IC50))

            adj_data = np.tile(single_adj_data,[self.test_x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[self.test_x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, self.test_x_mutation,self.test_x_expr,self.test_x_methylation]
            test_pred_value = self.predictor.predict(input_data)
            test_pred_value = test_pred_value.reshape(test_pred_value.shape[0])
            mean = np.mean(test_pred_value)
            std = np.std(test_pred_value)

            z_score = round((IC50-mean)/std,2)

            reward = self.reward(IC50,z_score)

            self.valid_count += 1
            self.total_reward += reward

            self.total_sample_list.append('<'+smi+'>'+str(round(IC50,4))+'>'+str(z_score))


            if reward > 1:
                self.winning_count += 1

        except:
            pass

        return IC50, z_score, reward

    def getReward(self,sample_list):
        total_pred_reward = 0
        valid_count = 0
        winning_count = 0
        bestReward = NO_REWARD
        bestSample = ""
        samples = []

        for sample in sample_list:

            IC50, z_score, reward = self.cal_reward(sample[1:-1])

            if reward > NO_REWARD:
                total_pred_reward += reward
                valid_count +=1
                #self.total_sample_list.append(sample+str(round(IC50,4))+'>'+str(z_score))

                if reward > 1:
                    samples.append(sample[1:-1])
                    winning_count += 1

                    self.update_que(reward, sample)


                if reward > bestReward:
                    bestReward = reward
                    bestSample = sample


        return total_pred_reward, valid_count, bestReward, bestSample, winning_count, samples




class MCTS():
    def __init__(self, explorationConstant=1 / math.sqrt(2), State=None,img_save_dir = './'
                 ,file_save_dir=None,self_train = False, GA = False, trainable = False):

        self.explorationConstant = explorationConstant
        self.root = TreeNode('<')
        self.State = State
        self.end_token = '>'
        self.img_save_dir = img_save_dir
        self.file_save_dir = file_save_dir
        self.root.winning_rate_list = []
        self.root.reward_rate_list = []
        self.loss = []
        self.root.nth = 0
        self.maxLength = 100

        self.self_train = self_train
        self.GA = GA
        self.trainable = trainable

    def load_dict(self,file_dir):
        loaded_dict = np.load(file_dir,allow_pickle='TRUE').item()
        importer = DictImporter()
        self.root = importer.import_(loaded_dict)

    def save_dict(self,save_dir):
        exporter = DictExporter()
        saved_dict = exporter.export(self.root)
        np.save(save_dir, saved_dict)

    def save_molecule_list(self,save_dir):
        unique_molecules = self.State.getUniqueList()
        data = []
        for i in unique_molecules:
            splited_i = i.split('>')
            smiles = splited_i[0][1:]
            IC50 = float(splited_i[1])
            z_score = float(splited_i[2])
            reward = self.State.reward(IC50,z_score)
            data.append([smiles,IC50,z_score,reward])


        df_data = pd.DataFrame(data=data,columns=['molecule','IC50','z_score','reward'])
        df_data = df_data.sort_values(by=['reward'],ascending = False)
        df_data.to_csv(save_dir)

    def getTreeImg(self,file_name):

        DotExporter(self.root,
                    nodenamefunc=lambda node: node.molecule,
                    edgeattrfunc=lambda parent, child: "style=bold,label=%0.3f" % (child.bestReward or 0)
        ).to_picture(self.img_save_dir+file_name)

    def getSampleMolecule(self,weight=2):
        node = self.root

        while (not node.isTerminal) and node.children:

            node_values = []
            for i in node.children:
                if i.bestReward != NO_REWARD:
                    value =  np.exp(weight * i.totalReward/i.valid_count)
                    node_values.append([i,value])
            try:
                df_node = pd.DataFrame(data= node_values, columns = ['node','value'])
                node = list(df_node.sample(n=1,weights='value')['node'])[0]
            except:
                break

        if node.isTerminal:
            return node.molecule

        elif node.bestReward != NO_REWARD:
            for i in range(100):
                sample = self.State.getSamples(current_molecule=node.molecule, num_samples = 1)
                if Chem.MolFromSmiles(sample[0][1:-1]):
                    return sample[0]

        return None


    def simulate(self,start=0,end=10000,numActions=10,dict_save_dir = '',
                 mode = 'CD&GD',stop = 10000):

        for self.root.nth in range(start,end+1):
            self.select(numActions=numActions)

            if self.self_train:
                try:
                    smis, rewards = self.State.self_train()
                    if len(smis) > 0:
                        [self.make_shortway(i[0],i[1]) for i in list(zip(smis,rewards))]
                        print('############### Num of Self trained samples ################ ', len(smis))
                except:
                    print("@@@@@@@@@@@ Error in self train @@@@@@@@@@@@@")

            if self.GA:
                try:
                    smis, rewards = self.State.GA()
                    if len(smis) > 0:
                        [self.make_shortway(i[0],i[1]) for i in list(zip(smis,rewards))]
                        print('############### Num of GA samples ################ ', len(smis))
                except:
                    print("@@@@@@@@@@@ Error in GA @@@@@@@@@@@@@")

            if self.trainable:
                try:
                    avg_loss = self.State.step()
                    self.loss.append(avg_loss)

                    print('############### avg_loss / queue size ################', avg_loss,
                                  ' / ', len(self.State.que.queue))
                except:
                    print("@@@@@@@@@@@ Error in step @@@@@@@@@@@@@")


            if (self.root.nth)%50 == 0 or self.State.valid_count >= stop:
                ## WR plot
                plt.plot(self.root.winning_rate_list)
                plt.xlabel(mode+': '+self.State.target_cell+' IC50 thres '+ str(self.State.thres_ic)+' z_score '
                           + str(self.State.thres_z)+' Simulation iteration')
                plt.ylabel('Winning rate')

                file_name =self.img_save_dir+'IC50_'+ str(self.State.thres_ic)+'_z_score_'\
                            + str(self.State.thres_z)+'_WR_'+str(self.root.nth)+'th.png'
                plt.savefig(file_name)
                plt.show()

                ## RR plot
                plt.plot(self.root.reward_rate_list)
                plt.xlabel(mode+': '+self.State.target_cell+' IC50 thres '+ str(self.State.thres_ic)+' z_score '
                               + str(self.State.thres_z)+' Simulation iteration')
                plt.ylabel('Reward rate')

                file_name =self.img_save_dir+'IC50_'+ str(self.State.thres_ic)+'_z_score_'\
                            + str(self.State.thres_z)+'_RR_'+str(self.root.nth)+'th.png'
                plt.savefig(file_name)

                plt.show()

                ## loss plot
                plt.plot(self.loss)
                plt.xlabel(mode+': '+self.State.target_cell+' IC50 thres '+ str(self.State.thres_ic)+' z_score '
                               + str(self.State.thres_z)+' Simulation iteration')
                plt.ylabel('Loss')

                file_name =self.img_save_dir+'IC50_'+ str(self.State.thres_ic)+'_z_score_'\
                            + str(self.State.thres_z)+'_Loss_'+str(self.root.nth)+'th.png'
                plt.savefig(file_name)

                plt.show()

                file_name = dict_save_dir+'_'+str(self.root.nth)+'th_dict.npy'
                self.save_dict(file_name)

                #file_name = '_'+str(self.nth)+'th_tree.png'
                #self.getTreeImg(file_name)

            #if self.root.nth%50 == 0 or self.State.valid_count >= stop:
                file_name = '_'+str(self.root.nth)+'th_output.csv'
                self.save_molecule_list(self.file_save_dir+file_name)

            WR = self.root.winning_count/self.root.valid_count
            print(self.root.nth,"th winning rate: ",WR)
            self.root.winning_rate_list.append(WR)
            RR = self.root.totalReward/self.root.valid_count
            self.root.reward_rate_list.append(RR)
            print(self.root.nth,"th reward rate: ",RR)
            print(self.root.nth,"th total valid num: ",self.State.valid_count)
            print(self.root.nth,"th total winning num: ",self.State.winning_count)

            if self.State.valid_count >= stop:
                break




    def select(self,numActions=10):
        parent = self.root
        children = parent.children
        while children != () and parent.depth < self.maxLength:
            child = self.getBestChild(parent)
            ## child is a terminal(None)
            if not child:
                break

            parent = child
            children = parent.children

        if (not parent.isTerminal):
            actions = []
            if parent.depth == self.maxLength-1 and not(parent.children):
                actions = [{'char':'>','prob':1}]
            elif parent.depth < self.maxLength-1:
                actions = self.getActions(parent,numSimul=numActions)

            for action in actions:
                self.expand(parent,action)

            if not actions:
                self.penalty(parent,valid_count=0)

        else:
            print(parent.molecule," is Terminal, depth: ",parent.depth)


    def getBestChild(self, node):
        bestValue = float("-inf")
        bestNodes = []
        randNum = random.randrange(0,20)
        if randNum == 0 and (node.bestReward != NO_REWARD) and node.depth < self.maxLength-1:
            sampleAction = self.getActions(node)[0]
            self.expand(node,sampleAction)

        for child in node.children:
            if child.name != self.end_token:

                try:
                    child.prob.device
                    child.prob = child.prob.detach().cpu().numpy()
                except:
                    pass

                actionValue = child.totalReward/child.valid_count
                nodeValue = actionValue \
                        + child.prob * self.explorationConstant * math.sqrt(child.parent.numVisits)/(child.numVisits+1)

                if nodeValue > bestValue:
                    bestValue = nodeValue
                    bestNodes = [child]
                elif nodeValue == bestValue:
                    bestNodes.append(child)

        if bestNodes != []:
            return random.choice(bestNodes)
        else:
            return None

    def getActions(self,node,numSimul=1):
        actions = self.State.getPossibleActions(node.molecule,numSimul=numSimul)
        return actions

    def getPredReward(self,molecule):
        sample_list = self.State.getSampleList(molecule)
        pred_reward, valid_count, bestReward, bestSample, winning_count, samples = self.State.getReward(sample_list)
        return pred_reward, valid_count, bestReward, bestSample, winning_count, samples

    def find_child(self,node,char):
        for child in node.children:
            if char == child.name:
                return child

        return None

    def make_shortway(self,molecule,reward):
        parent = self.root
        current = '<'
        for char in molecule[1:]:
            current += char
            child = self.find_child(parent,char)

            if not child:

                prob = self.State.generator.get_current_prob(self.State.data,
                                                             current)
                child = TreeNode(char, parent = parent, prob = prob)
                child.bestReward = reward
                child.bestSample = molecule
                child.samples.append(molecule)
            parent = child

            if parent.isTerminal:
                self.backpropogate(parent,reward,1,reward,molecule,1,[molecule])
                return True


    def expand(self,node,action):

        already_child = False
        for child in node.children:
            if action['char'] == child.name:
                if not(child.isTerminal):
                    already_child = True
                    break
                elif child.bestReward == NO_REWARD:
                    self.penalty(node,valid_count= 0)

                return

        if not already_child and node.bestReward != NO_REWARD:
            next_node = TreeNode(action['char'], parent = node, prob = action['prob'])

        else:
            next_node = node

        pred_reward, valid_count,bestReward,bestSample, winning_count, samples = self.getPredReward(next_node.molecule)

        if bestReward == NO_REWARD:
            self.penalty(next_node,valid_count = valid_count)

        else:
            self.backpropogate(next_node,pred_reward,valid_count,bestReward,bestSample,winning_count,samples)


    def node_scaffolds(self,node,ratio):
        frag_list = []
        for smiles in node.samples:
            try:
                mol = Chem.MolFromSmiles(smiles)
                frags = sg.get_all_murcko_fragments(mol)
                frags = [Chem.MolToSmiles(frag) for frag in frags if len(Chem.MolToSmiles(frag))>0]

                if len(frags)>0:
                    frag_list += frags

            except:
                pass

        count_scaf = dict(Counter(frag_list))

        node_scaf = []
        for key in count_scaf.keys():
            if count_scaf[key] > len(node.samples)*ratio:
                node_scaf.append(key)

        node.scaffolds = node_scaf

    def make_scaffold(self,threshold=0.5):
        for pre, _, node in RenderTree(self.root):
            self.node_scaffolds(node,ratio=threshold)


    def backpropogate(self, node, reward,valid_count,bestReward,bestSample,winning_count,samples):
        node.bestReward = bestReward
        node.bestSample = bestSample
        node.samples += samples
        node.samples = list(set(node.samples))

        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node.valid_count += valid_count
            node.winning_count += winning_count
            node = node.parent



    def penalty(self,node,valid_count=0):
        node.totalReward -= 1
        while node is not None:
            node.numVisits += 1
            node.valid_count += valid_count
            node = node.parent


class TreeNode(AnyNode):
    def __init__(self,name, parent=None,children=None,prob=0,
                 totalReward=1,bestReward=NO_REWARD,bestSample="",valid_count=1):
        super(TreeNode, self).__init__(parent,children)
        self.numVisits = 0
        self.prob = prob
        self.totalReward = 1
        self.bestReward = bestReward
        self.bestSample = bestSample
        self.valid_count = 1
        self.winning_count = 0
        self.name = name
        self.samples =[]
        self.scaffolds = []
        self.nth = 0

        if parent:
            self.molecule = parent.molecule + self.name
        else:
            self.molecule = self.name

        if self.name == '>':
            self.isTerminal = True
        else:
            self.isTerminal = False

        if not(parent):

            self.winning_rate_list = []
            self.reward_rate_list = []

class ScaffoldNode(AnyNode):
    def __init__(self,node=None,parent=None,children=None,WR=0,RR=0,scaffold="",point_dict ={},point = 0):
        super(ScaffoldNode, self).__init__(parent,children)
        self.node = node
        self.parent = parent
        self.drop = False
        self.WR = WR
        self.RR = RR
        self.scaffold = scaffold
        self.point_dict = point_dict
        self.point = point
        self.depth_point = 0

        if not point_dict:
            self.point_dict = {}
            for scaff in self.node.scaffolds:
                self.point_dict[scaff] = 0


class ScaffoldTree():
    def __init__(self,root=None,WR=0.5,RR=2):
        self.scaf_root = None
        self.turning_root = None
        self.node_depth_dict = {}
        self.node_list = []
        self.WR = WR
        self.RR = RR

        for pre, _, node in RenderTree(root):
            if len(node.scaffolds)>0 and node.winning_count/node.valid_count > WR \
                and node.totalReward/node.valid_count>RR:
                self.node_depth_dict[node] = node.depth

        self.node_depth_dict = dict(sorted(self.node_depth_dict.items(), key=operator.itemgetter(1),reverse=False))
        self.node_list = list(self.node_depth_dict.keys())

        WR = root.winning_count/root.valid_count
        RR = root.totalReward/root.valid_count

        scaffold = "<"

        self.scaf_root = ScaffoldNode(node = root,WR = WR, RR = RR,scaffold=scaffold,point_dict = None)


    def find_node(self, target_node,root=None):
        for pre, _, node in RenderTree(root):
            if node.node.molecule == target_node.molecule:
                return node
        return root


    def make_tree(self):

        temp_scaf_node = None
        target_node = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]

            depthest_node = self.scaf_root.node
            longest_molecule = ''

            for j in range(i-1,-1,-1):
                candi_node = self.node_list[j]
                parent = node.parent
                while (parent):
                    if parent.molecule == candi_node.molecule and len(longest_molecule) < len(parent.molecule):
                        depthest_node = candi_node
                        longest_molecule = candi_node.molecule

                    parent = parent.parent

            target_node = self.find_node(depthest_node,root = self.scaf_root)
            WR = node.winning_count/node.valid_count
            RR = node.totalReward/node.valid_count

            temp_scaf_node = ScaffoldNode(parent = target_node, node =node,
                                          WR = WR, RR = RR,point_dict = None)


    def make_scaffold_tree(self, threshold = 0.01):
        for pre, _, node in RenderTree(self.scaf_root):
            parent = node.parent
            while (parent):
                for parent_scaff in parent.node.scaffolds:
                    for node_scaff in node.node.scaffolds:
                        if parent_scaff == node_scaff:
                            parent.point_dict[parent_scaff] = parent.point_dict[parent_scaff]+1
                            parent.point += 1


                num_children = len(node.children)
                if num_children == 0:
                    for child in parent.children:
                        if not(set(node.node.scaffolds) - set(child.node.scaffolds))and\
                            child.node.molecule != node.node.molecule:
                            node.drop = True
                            break

                if not(set(node.node.scaffolds)-set(parent.node.scaffolds)):
                    node.drop = True

                parent = parent.parent

        self.node_depth_dict = {}
        self.node_list = []

        for pre, _, node in RenderTree(self.scaf_root):
            if not node.drop:
                self.node_depth_dict[node] = node.depth

        self.node_depth_dict = dict(sorted(self.node_depth_dict.items(), key=operator.itemgetter(1),reverse=False))
        self.node_list = list(self.node_depth_dict.keys())

        WR = self.scaf_root.WR
        RR = self.scaf_root.RR
        point = self.scaf_root.point

        self.turning_root = ScaffoldNode(node = self.scaf_root.node, scaffold=self.scaf_root.scaffold,
                                         WR = WR, RR = RR,point_dict=self.scaf_root.point_dict,point = point)

        temp_scaf_node = None
        target_node = None
        for i in range(len(self.node_list)):
            node = self.node_list[i].node

            depthest_node = self.turning_root.node
            longest_molecule = ''

            for j in range(i-1,-1,-1):
                candi_node = self.node_list[j].node
                parent = node.parent
                while (parent):
                    if parent.molecule == candi_node.molecule and len(longest_molecule) < len(parent.molecule):
                        depthest_node = candi_node
                        longest_molecule = candi_node.molecule


                    parent = parent.parent

            target_node = self.find_node(depthest_node,self.turning_root)

            node = self.node_list[i]
            point = node.point
            temp_scaf_node = ScaffoldNode(parent = target_node, node = node.node, scaffold= node.scaffold,
                                          WR = node.WR, RR = node.RR, point_dict = node.point_dict,
                                          point = point)

        for pre, _, node in RenderTree(self.turning_root):
            parent = node.parent

            if parent:
                best_node = None
                best_point = 0

                for brother in parent.children:
                    if not(set(node.node.scaffolds) - set(brother.node.scaffolds))and\
                        brother.node.molecule != node.node.molecule:

                        if best_point < brother.point:
                            best_node = brother
                            best_point = brother.point

                if best_node and best_point>node.point:
                    for scaff in node.point_dict.keys():
                        try:
                            best_node.point_dict[scaff] = best_node.point_dict[scaff]+node.point_dict[scaff]
                            best_node.point += node.point_dict[scaff]
                        except:
                            pass

                    for child in node.children:
                        child.parent = best_node


                    node.parent = None

        self.node_depth_dict = {}
        self.node_list = []

        for pre, _, node in RenderTree(self.turning_root):
            self.node_depth_dict[node] = node.depth

        self.node_depth_dict = dict(sorted(self.node_depth_dict.items(),
                                                key=operator.itemgetter(1),reverse=False))
        self.node_list = list(self.node_depth_dict.keys())[1:]

        depths = list(set(self.node_depth_dict.values()))[1:]

        i = 0
        for depth in depths:
            total_point = 1
            start=i
            while i < len(self.node_list) and self.node_list[i].depth == depth:
                 total_point += self.node_list[i].point
                 i +=1

            end = i
            for j in range(start,end):
                for scaff in self.node_list[j].point_dict.keys():

                    self.node_list[j].point_dict[scaff] =\
                        self.node_list[j].point_dict[scaff]/total_point*self.node_list[j].point

                self.node_list[j].depth_point = self.node_list[j].point/total_point

        for pre, _, node in RenderTree(self.turning_root):
            scaffold = ""
            sum_point = 0
            for scaff in node.node.scaffolds:
                scaffold += scaff+': '+str(round(node.point_dict[scaff],4))+'\n'
                sum_point += round(node.point_dict[scaff],4)
            node.scaffold = scaffold

            if node.depth_point < threshold:
                save = 0
                for child in node.children:
                    save+=self.check_child_node(child,save)
                if save ==0:
                    node.parent = None

        self.turning_root.scaffold = 'Root\n'

    def check_child_node(self,node,save):
        if node.depth_point > 0.2:
            return 1
        for child in node.children:
            self.check_child_node(child,save)
        return 0

    def getTreeImg(self,file_name):

        DotExporter(self.turning_root,
                    nodenamefunc=lambda node: node.scaffold\
                        +" RR: "+str(round(node.RR,2))+" WR: "+str(round(node.WR,2)),
                    edgeattrfunc=lambda parent, child: "style=bold,label=%0.3f" % (child.depth_point or 0)
        ).to_picture(file_name)
