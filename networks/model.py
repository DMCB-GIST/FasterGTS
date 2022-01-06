"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # num = int(bool(config.num_props)) +  int(config.scaffold_maxlen)    #  int(config.scaffold) 
        num = 1
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size + num, config.block_size + num))
                                     .view(1, 1, config.block_size + num, config.block_size + num))

        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        attn_save = att
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, attn_save

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        y, attn = self.attn(self.ln1(x))
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, attn

class ChemGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    
    # '$':2
    # '<':20
    # '>':22
    
    def __init__(self, config):
        super().__init__()

        # input embedding stem
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size-1, config.n_embd)
        self.pos_vec = torch.arange(start=0,end=config.block_size-1,dtype = int)
        
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size

        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.LSTM)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias') or ('bias' in pn):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        
        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        #self.pos_vec.to(idx.device)
        pos_mat = self.pos_vec.repeat(b,1).to(idx.device)
        
        
        position_embeddings = self.pos_emb(pos_mat)
        x = self.drop(token_embeddings + position_embeddings)
       
        for layer in self.blocks:
            x, attn = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        logits = logits[idx!=2]
        
        if targets is not None:
            targets = targets[idx!=2]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
    
    def generate(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        #self.pos_vec.to(idx.device)
        pos_mat = self.pos_vec[:t].repeat(b,1).to(idx.device)
        
        position_embeddings = self.pos_emb(pos_mat)
        x = (token_embeddings + position_embeddings)
       
        #attn_maps = []

        for layer in self.blocks:
            x, attn = layer(x)
         #   attn_maps.append(attn)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
    
    def get_samples(self,stoi,itos,n_samples,current_molecule='<'):
        self.eval()
        device = self.state_dict()['tok_emb.weight'].device
        len_mol = len(current_molecule)-1
        x = torch.tensor([stoi[i] for i in current_molecule], dtype=torch.long)[None,...].repeat(n_samples, 1).to(device)
        block_size = self.get_block_size()
        y = []
        for k in range(len_mol,100):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            logits = self.generate(x_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, ix), dim=1)
            if sum(x[:,-1] == stoi['>']):
                for i in x[(x[:,-1] == stoi['>'])]:
                    y.append(i)
                x = x[(x[:,-1] != stoi['>'])]
                
            if not len(x):
                break
        # for saving gpu memory
        probs = probs.detach().cpu().numpy()
        y = [i.detach().cpu().numpy() for i in y]
        samples = []
        for gen_mol in y:
            smi ='<'+''.join([itos[int(i)] for i in gen_mol])[1:-1]+'>'
            samples.append(smi)
    
        return samples
    
    def get_next_actions(self,stoi,itos,n_samples=1,current_molecule='<'):
        self.eval()
        device = self.state_dict()['tok_emb.weight'].device
        x = torch.tensor([stoi[i] for i in current_molecule], dtype=torch.long)[None,...].repeat(n_samples, 1).to(device)
        logits = self.generate(x)[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1).detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        action_list = [{'char':itos[ix[i].item()],'prob':probs[i][ix[i]]}  for i in range(len(ix))]
            
        return action_list
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
    
    def get_train_input(self, num_input = 20):
       
        # '$':2
        # '<':20
        # '>':22
        
        block_size = self.get_block_size()-1
        
        device = self.state_dict()['head.weight'].device
        x = torch.tensor([20], dtype=torch.long)[None,...].repeat(num_input, 1).to(device)
        tensor_2 = torch.tensor([2]).to(device)
        samples = []
        for k in range(block_size-1):
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
            logits = self.generate(x_cond)
            logits = logits[:, -1, :] #/ temperature
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            
            x = torch.cat((x, ix), dim=1)
            if sum(x[:,-1] == 22):
                for i in x[(x[:,-1] == 22)]:
                    samples.append(i)
                x = x[(x[:,-1] != 22)]
                
            if not len(x):
                break
        x = [torch.cat((i,tensor_2.repeat(block_size-len(i)))) for i in samples]
        x = torch.cat(x).reshape(num_input,block_size)
        y = [torch.cat((i[1:],tensor_2)) for i in x]
        y = torch.cat(y).reshape(num_input,block_size)
        
        return x, y
    
    def get_current_prob(self,stoi,current_molecule):
        
        device = self.state_dict()['tok_emb.weight'].device
        x = torch.tensor([stoi[i] for i in current_molecule[:-1]], dtype=torch.long)[None,...].repeat(1, 1).to(device)
        logits = self.generate(x)[:, -1, :]
        #probs = F.softmax(logits, dim=-1)
        inp = stoi[current_molecule[-1]]
        prob = torch.softmax(logits, dim=1).view(-1)[inp].cpu().detach().numpy()
        
        return prob
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

        