"""
This code is modified and added by Sejin Park, based on ReLeaSE (Reinforcement Learning for Structural Evolution).
The original code is from https://github.com/isayev/ReLeaSE.
Information about thier copyright is in https://github.com/isayev/ReLeaSE/blob/master/LICENSE.
"""

"""
This class implements generative recurrent neural network with augmented memory
stack as proposed in https://arxiv.org/abs/1503.01007
There are options of using LSTM or GRU, as well as using the generator without
memory stack.
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import time
from tqdm import trange

from utils import time_since

from smiles_enumerator import SmilesEnumerator

import numpy as np

import os


class StackAugmentedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_type='GRU',
                 n_layers=1, is_bidirectional=False, has_stack=False,
                 stack_width=None, stack_depth=None, use_cuda=None,
                 optimizer_instance=torch.optim.Adadelta, lr=0.01,gpuNum=0):

        super(StackAugmentedRNN, self).__init__()

        self.gpuNum = gpuNum
        if layer_type not in ['GRU', 'LSTM']:
            raise InvalidArgumentError('Layer type must be GRU or LSTM')
        self.layer_type = layer_type
        self.is_bidirectional = is_bidirectional
        if self.is_bidirectional:
            self.num_dir = 2
        else:
            self.num_dir = 1
        if layer_type == 'LSTM':
            self.has_cell = True
        else:
            self.has_cell = False
        self.has_stack = has_stack
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        if self.has_stack:
            self.stack_width = stack_width
            self.stack_depth = stack_depth

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        self.n_layers = n_layers

        if self.has_stack:
            self.stack_controls_layer = nn.Linear(in_features=self.hidden_size *
                                                              self.num_dir,
                                                  out_features=3)

            self.stack_input_layer = nn.Linear(in_features=self.hidden_size *
                                                           self.num_dir,
                                               out_features=self.stack_width)

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.has_stack:
            rnn_input_size = hidden_size + stack_width
        else:
            rnn_input_size = hidden_size
        if self.layer_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_size, hidden_size, n_layers,
                               bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        elif self.layer_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_size, hidden_size, n_layers,
                             bidirectional=self.is_bidirectional)
            self.decoder = nn.Linear(hidden_size * self.num_dir, output_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

        if self.use_cuda:
            torch.cuda.set_device(self.gpuNum)
            self = self.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer_instance = optimizer_instance
        self.optimizer = self.optimizer_instance(self.parameters(), lr=lr,
                                                 weight_decay=0.00001)

    def load_model(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def change_lr(self, new_lr):

        self.optimizer = self.optimizer_instance(self.parameters(), lr=new_lr)
        self.lr = new_lr

    def forward(self, inp, hidden, stack):
        inp = self.encoder(inp.view(1, -1))
        if self.has_stack:
            if self.has_cell:
                hidden_ = hidden[0]
            else:
                hidden_ = hidden
            if self.is_bidirectional:
                hidden_2_stack = torch.cat((hidden_[0], hidden_[1]), dim=1)
            else:
                hidden_2_stack = hidden_.squeeze(0)
            stack_controls = self.stack_controls_layer(hidden_2_stack)
            stack_controls = F.softmax(stack_controls, dim=1)
            stack_input = self.stack_input_layer(hidden_2_stack.unsqueeze(0))
            stack_input = torch.tanh(stack_input)
            stack = self.stack_augmentation(stack_input.permute(1, 0, 2),
                                            stack, stack_controls)
            stack_top = stack[:, 0, :].unsqueeze(0)
            inp = torch.cat((inp, stack_top), dim=2)
        output, next_hidden = self.rnn(inp.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, next_hidden, stack

    def stack_augmentation(self, input_val, prev_stack, controls):

        batch_size = prev_stack.size(0)

        controls = controls.view(-1, 3, 1, 1)
        zeros_at_the_bottom = torch.zeros(batch_size, 1, self.stack_width)
        if self.use_cuda:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom.cuda())
        else:
            zeros_at_the_bottom = Variable(zeros_at_the_bottom)
        a_push, a_pop, a_no_op = controls[:, 0], controls[:, 1], controls[:, 2]
        stack_down = torch.cat((prev_stack[:, 1:], zeros_at_the_bottom), dim=1)
        stack_up = torch.cat((input_val, prev_stack[:, :-1]), dim=1)
        new_stack = a_no_op * prev_stack + a_push * stack_up + a_pop * stack_down
        return new_stack

    def init_hidden(self):

        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_cell(self):

        if self.use_cuda:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size).cuda())
        else:
            return Variable(torch.zeros(self.n_layers * self.num_dir, 1,
                                        self.hidden_size))

    def init_stack(self):

        result = torch.zeros(1, self.stack_depth, self.stack_width)
        if self.use_cuda:
            return Variable(result.cuda())
        else:
            return Variable(result)

    def train_step(self, inp, target):

        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        self.optimizer.zero_grad()
        loss = 0
        for c in range(len(inp)):
            output, hidden, stack = self(inp[c], hidden, stack)
            loss += self.criterion(output, target[c].unsqueeze(0))
        #print("loss ",loss.item())
        loss.backward()
        self.optimizer.step()

        return loss.item() / len(inp)
    
    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out


    def evaluate(self, data, prime_str='<', end_token='>', predict_len=100):

        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None
        prime_input = data.char_tensor(prime_str)
        new_sample = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str)-1):
            _, hidden, stack = self.forward(prime_input[p], hidden, stack)
        inp = prime_input[-1]


        for p in range(predict_len):
            output, hidden, stack = self.forward(inp, hidden, stack)
            output = self.top_k_logits(output, 20)
            
            # Sample from the network as a multinomial distribution
            probs = torch.softmax(output, dim=1)
            
            top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

            # Add predicted character to string and use as next input
            predicted_char = data.all_characters[top_i]
            new_sample += predicted_char
            inp = data.char_tensor(predicted_char)
            if predicted_char == end_token:
                break
        return new_sample

    def fit(self, data, n_iterations, all_losses=[], print_every=100,
            plot_every=10, augment=False):

        start = time.time()
        loss_avg = 0

        if augment:
            smiles_augmentation = SmilesEnumerator()
        else:
            smiles_augmentation = None

        for epoch in trange(1, n_iterations + 1, desc='Training in progress...'):
            inp, target = data.random_training_set(smiles_augmentation)
            loss = self.train_step(inp, target)
            loss_avg += loss

            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch,
                                               epoch / n_iterations * 100, loss)
                      )
                print(self.evaluate(data=data, prime_str = '<',
                                    predict_len=100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
        return all_losses

    """
    From here,
    written by Sejin Park.
    """
    def get_current_weight(self, data, current_molecule = '<' ,end_token='>', predict_len=100):

        output = None
        stack = None
        hidden = self.init_hidden()
        if self.has_cell:
            cell = self.init_cell()
            hidden = (hidden, cell)
        if self.has_stack:
            stack = self.init_stack()
        else:
            stack = None

        for char in current_molecule:
            inp = data.char_tensor(char)
            output, hidden, stack = self.forward(inp, hidden, stack)

        return output, hidden, stack
    
    def get_current_prob(self,data,current_molecule):
        output, _, _ = self.get_current_weight(data,current_molecule[:-1])
        inp = data.char_tensor(current_molecule[-1])
        prob = torch.softmax(output, dim=1).view(-1)[inp].cpu().detach().numpy()
        
        return prob
        
    
    def get_next_actions(self, data, current_molecule = '<' ,end_token='>', predict_len=100,numActions=10):
        output, hidden, stack = self.get_current_weight(data,current_molecule)

        action_list = []

        #Sample from the network as a multinomial distribution
        probs = torch.softmax(output, dim=1)
        #top_i = torch.multinomial(probs.view(-1), rank_range, replacement = self.replacement).cpu().numpy()

        del hidden
        del stack
        del output
        torch.cuda.empty_cache()
        
        top_i = list(set(torch.multinomial(probs.view(-1), numActions, replacement = True).cpu().numpy()))
        for i in top_i:
            predicted_char = data.all_characters[i]
            prob = probs[0][i].cpu().detach().numpy()
            action_list.append({'char':predicted_char,'prob':prob})
            
        del probs
        return action_list

    def new_simulation(self, data,current_molecule='<', end_token='>',numSimul = 10, predict_len=100):
        output, hidden, stack = self.get_current_weight(data,current_molecule)

        inp = data.char_tensor(current_molecule[-1])
        sample_list = []
        for i in range(numSimul):
            temp_hidden = hidden
            temp_stack = stack
            temp_inp = inp
            new_sample = current_molecule

            for p in range(len(current_molecule),predict_len+1):
                output, temp_hidden, temp_stack = self.forward(temp_inp, temp_hidden, temp_stack)

                # Sample from the network as a multinomial distribution
                probs = torch.softmax(output, dim=1)
                top_i = torch.multinomial(probs.view(-1), 1)[0].cpu().numpy()

                # Add predicted character to string and use as next input
                predicted_char = data.all_characters[top_i]
                new_sample += predicted_char
                temp_inp = data.char_tensor(predicted_char)
                if predicted_char == end_token:
                    break
                elif p == predict_len:
                    new_sample = new_sample[:-1]+'>'

            sample_list.append(new_sample)

        del hidden
        del stack

        return sample_list
