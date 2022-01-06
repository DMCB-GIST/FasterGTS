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
import numpy as np
from rdkit import Chem
from torch.autograd import Variable
import pandas as pd
import random

class Reinforcement(object):
    def __init__(self, generator, predictor, get_reward, adj_fp_fun, x_mutation=None,x_expr=None,x_methylation=None):

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



    def policy_gradient(self, data, n_batch=10, gamma=0.97,
                        grad_clipping=None, **kwargs):
        rl_loss = 0
        self.generator.optimizer.zero_grad()
        total_reward = 0

        for _ in range(n_batch):

            # Sampling new trajectory
            reward = 0
            trajectory = '<>'
            while reward == 0:
                trajectory = self.generator.evaluate(data)
                mol = Chem.MolFromSmiles(trajectory[1:-1])

                if mol and len(trajectory)>2:
                    random_index = random.randrange(0,len(self.x_mutation))
                    reward = self.get_reward(trajectory[1:-1],
                                             self.predictor,
                                             self.adj_fp_fun,
                                             self.x_mutation[random_index:(random_index+1)],
                                             self.x_expr[random_index:(random_index+1)],
                                             self.x_methylation[random_index:(random_index+1)])
                #else:
                #    reward = 0

            # Converting string of characters into tensor
            trajectory_input = data.char_tensor(trajectory)
            discounted_reward = reward
            total_reward += reward

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
            for p in range(len(trajectory)-1):
                output, hidden, stack = self.generator(trajectory_input[p],
                                                       hidden,
                                                       stack)
                log_probs = F.log_softmax(output, dim=1)
                top_i = trajectory_input[p+1]
                rl_loss -= ((log_probs[0, top_i])*discounted_reward)
                discounted_reward = discounted_reward * gamma

        # Doing backward pass and parameters update
        rl_loss = rl_loss / n_batch
        total_reward = total_reward / n_batch
        rl_loss.backward()
        if grad_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                           grad_clipping)

        self.generator.optimizer.step()

        return total_reward, rl_loss.item()
