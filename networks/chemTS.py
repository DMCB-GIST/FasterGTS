import numpy as np
from math import *
import matplotlib.pyplot as plt
import pandas as pd
import random as pr

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem


class chemical:

    def __init__(self):

        self.position='<' #start symbol
    def Clone(self):

        st = chemical()
        st.position= self.position
        return st

    def SelectPosition(self,m):

        self.position+=m

    def Getatom(self):
        return [i for i in range(self.num_atom)]

class Node:

    def __init__(self, position = None,  parent = None, state = None):
        self.position = position
        self.parentNode = parent
        self.childNodes = []
        self.child = []
        self.wins = 0
        self.visits = 0
        self.depth=0
        self.end_token = '>'


    def Selectnode(self):
        ucb=[]
        for i in range(len(self.childNodes)):
            ucb.append(self.childNodes[i].wins/self.childNodes[i].visits+1.0\
                       *sqrt(2*log(self.visits)/self.childNodes[i].visits))
                
        m = np.amax(ucb)
        indices = np.nonzero(ucb == m)[0]
        ind=pr.choice(indices)
        s=self.childNodes[ind]
        return s

    def Addnode(self, m, s):
        n = Node(position = m, parent = self, state = s)
        self.childNodes.append(n)
        self.child.append(m)
        return n
    
    def get_info(self):
        return self.visits, self.wins
    
    def Update(self, result):

        self.visits += 1
        self.wins += result #reward
        
    def backup(self,visits,wins):
        self.visits += visits
        self.wins += wins
        
    def check_child(self,m):
        try:
            index = self.child.index(m)
            
            return self.childNodes[index]
        except:
            return False

def reward_fun(alpha,beta,IC50,z_score,thres_ic,thres_z):
    if z_score <= thres_z and IC50 <= thres_ic:
        return np.exp(alpha*(-z_score+thres_z))+beta*np.log(-IC50+thres_ic+1)
    else:
        return 1.


def getReward(sample_list,all_node_indices,adj_fp_fun,x_mutation,x_expr,x_methylation,
              test_x_mutation,test_x_expr,test_x_methylation, predictor,
              thres_ic, thres_z,alpha,beta,total_sample_list):
    
    total_pred_reward = []
    valid_samples = []
    valid_node_indices = []
    winning_count = 0
    for sample in sample_list:
        try:
            mol = Chem.MolFromSmiles(sample[1:-1])
            
            adj_fp_data =  adj_fp_fun.featurize([mol])
            single_adj_data = adj_fp_data[0][0].reshape(1,100,100)
            single_drug_feat_data = adj_fp_data[0][1].reshape(1,100,75)
            adj_data = np.tile(single_adj_data,[x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, x_mutation,x_expr,x_methylation]

            IC50 = predictor.predict(input_data)
            IC50 = np.average(list(IC50))
            #print("IC50 ",IC50)

            adj_data = np.tile(single_adj_data,[test_x_mutation.shape[0],1,1])
            drug_feat_data = np.tile(single_drug_feat_data,[test_x_mutation.shape[0],1,1])

            input_data = [drug_feat_data,adj_data, test_x_mutation,test_x_expr,test_x_methylation]

            test_pred_value = predictor.predict(input_data)
            test_pred_value = test_pred_value.reshape(test_pred_value.shape[0])
            mean = np.mean(test_pred_value)
            std = np.std(test_pred_value)
            z_score = round((IC50-mean)/std,2)

            #print("z_score ",z_score)
            temp_reward = 1
            if IC50 <= thres_ic and z_score <= thres_z:
                winning_count +=1
                temp_reward = reward_fun(alpha,beta,IC50,z_score,thres_ic,thres_z)
            
            total_pred_reward.append(temp_reward)
            total_sample_list.append(sample+str(round(IC50,4))+'>'+str(z_score))
            
            valid_samples.append(sample[1:-1])
            valid_node_indices.append(all_node_indices[sample_list.index(sample)])
                
        except:
            pass

    return total_pred_reward, valid_samples, valid_node_indices, winning_count


def expanded_node(model,data, state_position):
    
    action_list = model.get_next_actions(data, current_molecule=state_position,numActions=10)
    all_nodes = [i['char'] for i in action_list]
    
    return all_nodes

def simulation(model,data,state_position,all_nodes,numSample=5):
    all_posible = [] 
    node_indices = []
    
    for next_char in all_nodes:
        temp_state_molecule = state_position+next_char
        temp_samples = model.new_simulation(data, current_molecule= temp_state_molecule, numSimul = numSample)
        temp_samples = list(set(temp_samples))
        all_posible += temp_samples
        node_indices += [all_nodes.index(next_char)]*len(temp_samples)
        
    
    return all_posible, node_indices

def save_molecule_list(total_sample_list,alpha,beta,thres_IC50,thres_z_score,file_name,save_dir):
    unique_molecules = list(set(total_sample_list))
    data = []
    for i in unique_molecules:
        #try:
        splited_i = i.split('>')
        smiles = splited_i[0][1:]
        IC50 = float(splited_i[1])
        z_score = float(splited_i[2])
        reward = reward_fun(alpha,beta,IC50,z_score,thres_IC50,thres_z_score)
        data.append([smiles,IC50,z_score,reward])
        #except:
        #    pass

    df_data = pd.DataFrame(data=data,columns=['molecule','IC50','z_score','reward'])
    df_data = df_data.sort_values(by=['reward'],ascending = False)
    df_data.to_csv(save_dir+file_name)

def plot_WW_RR(winning_rate_list,reward_rate_list,thres_ic,thres_z, target_cell,img_save_dir):
    #nth = str(len(winning_rate_list))
    ## WR plot
    
    plt.plot(winning_rate_list)
    plt.xlabel(target_cell+' IC50 thres '+ str(thres_ic)+' z_score '
                           + str(thres_z)+' Simulation iteration')
    plt.ylabel('Winning rate')

    file_name = img_save_dir+target_cell+'_IC50_'+ str(thres_ic)+'_z_score_'\
                + str(thres_z)+'_WR.png'
    plt.savefig(file_name)
    plt.show()

    ## RR plot
    plt.plot(reward_rate_list)
    plt.xlabel(target_cell+' IC50 thres '+ str(thres_ic)+' z_score '
                               + str(thres_z)+' Simulation iteration')
    plt.ylabel('Reward rate')

    file_name =img_save_dir+target_cell+'_IC50_'+ str(thres_ic)+'_z_score_'\
                            + str(thres_z)+'_RR.png'
    plt.savefig(file_name)

    plt.show()

    #file_name = dict_save_dir+'_'+nth+'th_dict.npy'
    #self.save_dict(file_name)

    #file_name = '_'+str(self.root.nth)+'th_output.csv'
    #self.save_molecule_list(self.file_save_dir+file_name)
    

def MCTS(root, model, target_cell, rate_save_dir ,img_save_dir, results_save_dir, adj_fp_fun,
         x_mutation,x_expr,x_methylation, test_x_mutation,test_x_expr,test_x_methylation,
         predictor,thres_ic, thres_z,alpha,beta, data, max_iter = 1000):

    """initialization of the chemical trees and grammar trees"""
    rootnode = Node(state = root)
    state = root.Clone()
    it=0
    """----------------------------------------------------------------------"""


    """global variables used for save valid compounds and simulated compounds"""
    valid_compound=[]
    all_simulated_compound=[]
    max_score=-100.0
    current_score=[]
    #depth=[]
    
    total_sample_list = []
    reward_list = []
    total_winning_rate = []
    total_reward_rate = []
    
    total_winning_count = 0.
    total_reward = 0.
    total_valid_count = 0.

    while it < max_iter:
        node = rootnode
        state = root.Clone()
        """selection step"""
        node_pool=[]
        print("current found max_score:",max_score)
        print(str(it+1)+"th Nub Valid sampling:",len(valid_compound))
        
        while node.childNodes!=[]:
            node = node.Selectnode()
            state.SelectPosition(node.position)
            
        print("state position:,",state.position)
        #depth.append(len(state.position))
        
        if len(state.position)>101:
            re=-1.0
            while node != None:
                node.Update(re)
                node = node.parentNode
        else:
            it += 1
            all_nodes=expanded_node(model,data, state.position)
           
            all_posible, all_node_indices =simulation(model,data,state.position,all_nodes,numSample=30)
            score, valid_samples,valid_node_indices, winning_count = getReward(all_posible,all_node_indices,adj_fp_fun,
                                                                                x_mutation,x_expr,x_methylation,
                                                                                test_x_mutation,test_x_expr,test_x_methylation,
                                                                                predictor,thres_ic, thres_z,
                                                                                alpha,beta,total_sample_list)
                            
            valid_compound.extend(valid_samples)
            all_simulated_compound.extend(all_posible)
            reward_list.extend(score)
            total_valid_count += len(valid_samples)
            total_winning_count += winning_count
            total_reward += sum(score)
            
            total_winning_rate.append(total_winning_count/total_valid_count)
            total_reward_rate.append(total_reward/total_valid_count)
            
            if len(valid_samples)==0:
                re=-1.0
                while node != None:
                    node.Update(re)
                    node = node.parentNode
            else:
                #re=[]
                for i in range(len(valid_samples)):
                    m=valid_node_indices[i]
                    re = (0.8*score[i])/(1.0+abs(0.8*score[i]))
                    
                    child_node = node.check_child(all_nodes[m])
                    if not child_node:
                        child_node = node.Addnode(all_nodes[m],state)
                        node_pool.append(child_node)
                    
                    child_node.Update(re)
                    #else:
                        
                    if score[i]>=max_score:
                        max_score=score[i]
                        current_score.append(max_score)
                    else:
                        current_score.append(max_score)
                        
                    #depth.append(len(state.position))
                        
                
                for i in range(len(node_pool)):
                    node=node_pool[i]
                    visits, wins = node.get_info()
                    node = node.parentNode
                    while node != None:
                        #re = node.Update(re[i])
                        node.backup(visits, wins)
                        node = node.parentNode
                        
              
            if (it+1) % 500 == 0:
                plot_WW_RR(total_winning_rate,total_reward_rate,thres_ic,thres_z, target_cell, img_save_dir)
                file_name = target_cell+'_'+str(it)+'th_ChemTS_results.csv'
                save_molecule_list(total_sample_list,alpha,beta,thres_ic,thres_z, file_name, results_save_dir)
                pd.DataFrame(data=total_reward_rate).to_csv(rate_save_dir+target_cell+'_'+str(it)+'th_reward.csv',index=None)
                pd.DataFrame(data=total_winning_rate).to_csv(rate_save_dir+target_cell+'_'+str(it)+'th_winning.csv',index=None)
                
    return valid_compound, total_sample_list, rootnode