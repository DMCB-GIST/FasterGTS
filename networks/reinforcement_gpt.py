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


class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward, adj_fp_fun, stoi=None,x_mutation=None,x_expr=None,x_methylation=None):

        super(Reinforcement, self).__init__()
        self.generator = generator
        self.predictor = predictor
        self.get_reward = get_reward

        self.adj_fp_fun = adj_fp_fun
        self.x_mutation = x_mutation
        self.x_expr = x_expr
        self.x_methylation = x_methylation

        self.mutation_list = []
        self.expr_list = []
        self.methylation_list = []
        self.stoi = stoi


    def policy_gradient(self, samples,optimizer,scaler, device, gamma=0.97,
                        grad_norm_clip = None, **kwargs):
        self.generator.train()
        rl_loss = 0
        total_reward = 0
        num_samples = float(len(samples))
        
        for smi in samples:

            # Sampling new trajectory
            reward = 0
            
            random_index = random.randrange(0,len(self.x_mutation))
            reward = self.get_reward(smi,
                                     self.predictor,
                                     self.adj_fp_fun,
                                     self.x_mutation[random_index:(random_index+1)],
                                     self.x_expr[random_index:(random_index+1)],
                                     self.x_methylation[random_index:(random_index+1)])
                
            discounted_reward = reward
            total_reward += reward
            input_tensor = torch.tensor([self.stoi[i] for i in smi])#.to(device)
                
            for k in range(len(input_tensor)-1):
                x_cond = input_tensor[:(k+1)].reshape(1,k+1).to(device)
                logits = self.generator.generate(x_cond)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=1)
                
                top_i = input_tensor[k+1].item()
                rl_loss -= ((log_probs[0, top_i])*discounted_reward)
                discounted_reward = discounted_reward * gamma
           
        # Doing backward pass and parameters update
       
        if total_reward > 0:
            rl_loss = rl_loss / num_samples
            total_reward = total_reward / num_samples
            
            optimizer.zero_grad()
            scaler.scale(rl_loss).backward()
            if grad_norm_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()
            print(rl_loss)
            
        self.generator.eval()
        return total_reward, rl_loss.item()
