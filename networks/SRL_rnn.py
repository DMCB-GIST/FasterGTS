"""
This code is modified and added by Sejin Park, based on ReLeaSE (Reinforcement Learning for Structural Evolution).
The original code is from https://github.com/isayev/ReLeaSE.
Information about thier copyright is in https://github.com/isayev/ReLeaSE/blob/master/LICENSE.
"""

"""
This class implements simple policy gradient algorithm for
biasing the generation of molecules towards desired values of
properties aka Reinforcement Learninf for Structural Evolution (ReLeaSE)
as described in
Popova, M., Isayev, O., & Tropsha, A. (2018).
Deep reinforcement learning for de novo drug design.
Science advances, 4(7), eaap7885.
"""
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem


def simple_moving_average(previous_values, new_value, ma_window_size=10):
    value_ma = np.sum(previous_values[-(ma_window_size-1):]) + new_value
    value_ma = value_ma/(len(previous_values[-(ma_window_size-1):]) + 1)
    return value_ma

def generate_samples(generator,gen_data,n_samples):
    generator.eval()
    samples = []
    for i in range(n_samples):
        smi = generator.evaluate(gen_data, predict_len=120)[1:-1]
        mol = Chem.MolFromSmiles(smi)
        if mol and len(smi)>3:
            samples.append(smi)

    return samples


def train(RL_max, generator,gen_data,target_cell,
          n_to_generate,n_policy,n_try,save_dir, img_save_dir):

    rewards_max = []
    rl_losses_max = []

    valid_ratio_list = []
    unique_ratio_list = []


    for i in range(n_policy):
        samples = []
        while len(samples)==0:
            samples = list(set(generate_samples(generator,gen_data,n_to_generate)))


        cur_reward, cur_loss = RL_max.policy_gradient(samples)
        rewards_max.append(simple_moving_average(rewards_max, cur_reward))
        rl_losses_max.append(simple_moving_average(rl_losses_max, cur_loss))

        test_samples = generate_samples(generator,gen_data,n_try)
        valid_ratio = len(test_samples)/n_try
        unique_ratio = len(list(set(test_samples)))/n_try

        valid_ratio_list.append(valid_ratio)
        unique_ratio_list.append(unique_ratio)


        plt.plot(rewards_max)
        plt.xlabel(target_cell+' Training iteration')
        plt.savefig(img_save_dir+target_cell+'_reward.png')
        plt.show()

        plt.plot(rl_losses_max)
        plt.xlabel(target_cell+' Training iteration')
        plt.ylabel('Loss')
        plt.savefig(img_save_dir+target_cell+'_loss.png')
        plt.show()

        plt.plot(valid_ratio_list,label = 'valid')
        plt.plot(unique_ratio_list,label = 'unique')
        plt.legend()
        plt.xlabel(target_cell+' valid/unique_ratio')
        plt.ylabel('ratio')
        plt.savefig(img_save_dir+target_cell+'_valid_unique_ratio.png')
        plt.show()



        RL_max.save_molecule_list(save_dir)
        RL_max.plot_WW_RR(target_cell,img_save_dir)
        #pd.DataFrame(data=RL_max.total_reward_rate).to_csv(rate_save_dir+target_cell+'_reward.csv',index=None)
        #pd.DataFrame(data=RL_max.total_winning_rate).to_csv(rate_save_dir+target_cell+'_winning.csv',index=None)

        if valid_ratio <= 0.3 or unique_ratio <= 0.2:
            break;

    prl_weight = '/FasterGTS/pre_trained_weights/PRL_rnn_'+target_cell+'.pt'
    torch.save(generator.state_dict(),prl_weight)

    return rewards_max, rl_losses_max

class Personalized_RL(object):
    def __init__(self, generator, predictor, adj_fp_fun, data = None,
                 x_mutation=None,x_expr=None,x_methylation=None,
                 test_x_mutation=None,test_x_expr=None,test_x_methylation=None,
                 alpha=0,beta=0,thres_ic=0,thres_z=0):

        super(Personalized_RL, self).__init__()
        self.generator = generator
        self.predictor = predictor

        self.data = data

        self.adj_fp_fun = adj_fp_fun

        self.x_mutation = x_mutation
        self.x_expr = x_expr
        self.x_methylation = x_methylation

        self.test_x_mutation = test_x_mutation
        self.test_x_expr = test_x_expr
        self.test_x_methylation = test_x_methylation

        self.mutation_list = []
        self.expr_list = []
        self.methylation_list = []

        self.alpha = alpha
        self.beta = beta
        self.thres_ic = thres_ic
        self.thres_z = thres_z

        self.total_sample_list = []

        self.total_winning_count = 0.
        self.total_valid_count = 0.
        self.total_reward = 0.

        self.total_reward_rate = []
        self.total_winning_rate = []

    def reward_fun(self,IC50,z_score):
        if z_score <= self.thres_z and IC50 <= self.thres_ic:
            return np.exp(self.alpha*(-z_score+self.thres_z))+self.beta*np.log(-IC50+self.thres_ic+1)
        else:
            return 1.

    def get_reward(self,smi):
        reward = 1.

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
            print("IC50 ",IC50)

            adj_data = np.tile(single_adj_data,[self.test_x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[self.test_x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, self.test_x_mutation,self.test_x_expr,self.test_x_methylation]

            test_pred_value = self.predictor.predict(input_data)
            test_pred_value = test_pred_value.reshape(test_pred_value.shape[0])
            mean = np.mean(test_pred_value)
            std = np.std(test_pred_value)
            z_score = round((IC50-mean)/std,2)

            print("z_score ",z_score)
            if IC50 <= self.thres_ic and z_score <= self.thres_z:
                self.total_winning_count +=1
                reward = self.reward_fun(IC50,z_score)
                self.total_reward += reward

            self.total_valid_count += 1

            self.total_sample_list.append('<'+smi+'>'+str(round(IC50,4))+'>'+str(z_score))

        except:
            pass

        return reward

    def policy_gradient(self, samples, gamma=0.97,
                        grad_norm_clip = None, **kwargs):
        self.generator.train()

        self.generator.optimizer.zero_grad()
        rl_loss = 0.
        sum_reward = 0.
        avg_reward = 0.
        num_samples = float(len(samples))

        for smi in samples:

            # Sampling new trajectory
            reward = self.get_reward(smi)

            discounted_reward = reward
            sum_reward += reward

            trajectory_input = self.data.char_tensor(smi)

            # Initializing the generator's hidden state
            hidden = self.generator.init_hidden()
            if self.generator.has_cell:
                cell = self.generator.init_cell()
                hidden = (hidden, cell)
            if self.generator.has_stack:
                stack = self.generator.init_stack()
            else:
                stack = None

            # "Following" the trajectory and accumulating the loss
            for p in range(len(smi)-1):
                output, hidden, stack = self.generator(trajectory_input[p],
                                                       hidden,
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= ((log_probs[0, top_i])*discounted_reward)
                discounted_reward = discounted_reward * gamma


        # Doing backward pass and parameters update
        self.total_reward_rate.append(self.total_reward/self.total_valid_count)
        self.total_winning_rate.append(self.total_winning_count/self.total_valid_count)

        #if sum_reward > 0:
        rl_loss = rl_loss / num_samples
        avg_reward = sum_reward / num_samples

        rl_loss.backward()
        if grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                           grad_norm_clip)
        self.generator.optimizer.step()

        #else:
        #    rl_loss = 0
        #    avg_reward = 0
        self.generator.eval()

        return avg_reward, rl_loss.item()

    def save_molecule_list(self,save_dir):
        unique_molecules = list(set(self.total_sample_list))
        data = []
        for i in unique_molecules:
            try:
                splited_i = i.split('>')
                smiles = splited_i[0][1:]
                IC50 = float(splited_i[1])
                z_score = float(splited_i[2])
                reward = self.reward_fun(IC50,z_score)
                data.append([smiles,IC50,z_score,reward])
            except:
                pass

        df_data = pd.DataFrame(data=data,columns=['molecule','IC50','z_score','reward'])
        df_data = df_data.sort_values(by=['reward'],ascending = False)
        df_data.to_csv(save_dir)

    def plot_WW_RR(self,target_cell,img_save_dir):

        plt.plot(self.total_winning_rate)
        plt.xlabel(target_cell+' IC50 thres '+ str(self.thres_ic)+' z_score '
                               + str(self.thres_z)+' Simulation iteration')
        plt.ylabel('Winning rate')

        file_name = img_save_dir+target_cell+'_IC50_'+ str(self.thres_ic)+'_z_score_'\
                    + str(self.thres_z)+'_WR.png'
        plt.savefig(file_name)
        plt.show()

        ## RR plot
        plt.plot(self.total_reward_rate)
        plt.xlabel(target_cell+' IC50 thres '+ str(self.thres_ic)+' z_score '
                                   + str(self.thres_z)+' Simulation iteration')
        plt.ylabel('Reward rate')

        file_name =img_save_dir+target_cell+'_IC50_'+ str(self.thres_ic)+'_z_score_'\
                                + str(self.thres_z)+'_RR.png'
        plt.savefig(file_name)

        plt.show()
