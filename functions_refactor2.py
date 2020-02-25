#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 13:59:51 2020

@author: will
"""


import torch
import numpy as np
from torch.utils import data

class DatasetTxts(data.Dataset):
    # special_tokens should be a list [torch.tensor([cls_token_id]), torch.tensor([sep_token_id])]
    # datalist is something like list(zip(q_title,q_body,answer)), list(zip(answer,))
    # budget is a list like [0.2, 0.4,0.4]
    def __init__(self, datalist, max_len, special_tokens, budget, dtype, yhat=None, aug=None, token=None):
        self.datalist = datalist
        self.budget = np.array(budget)
        self.yhat = yhat
        self.aug = aug
        self.token = token
        self.special_tokens = special_tokens
        self.n_item = len(datalist[0])
        self.max_len = max_len - len(special_tokens[0]) - len(special_tokens[1]) * self.n_item
        self.dtype = dtype
        
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = self.datalist[index]
        if self.aug is not None and self.token is not None:
            data = [self.token.encode(self.aug(i),add_special_tokens=False) for i in data]
        budget_left = np.array([p*self.max_len - len(d) for p,d in zip(self.budget,data)])
        order = np.argsort(-budget_left)
        reverse_order = np.argsort(order)
        
        # switch order
        data = self.switch_order(data,order)
        budget = self.switch_order(self.budget,order)
        
        budget = self.update_budget(budget)
        tot_len = self.max_len
        for i in range(self.n_item):
            data[i] = self.truncate(data[i],int(tot_len*budget[i]),self.dtype)
            tot_len = tot_len - len(data[i])
        
        data = self.switch_order(data,reverse_order)
        return self.add_special_tokens(data), self.yhat[index] if self.yhat is not None else None
        
    @staticmethod
    def switch_order(data,order):
        return [data[i] for i in order]
    
    @staticmethod
    def update_budget(budget):
        # compute conditional budget for remainin item
        return budget/np.cumsum(budget[::-1])[::-1]
    
    @staticmethod
    def truncate(x, max_len, dtype):
        if len(x) <= max_len:
            return torch.tensor(x,dtype=dtype)
        else:
            start = np.random.randint(0,len(x)-max_len+1)
            return torch.tensor(x[start:start+max_len],dtype=dtype)
    
    def add_special_tokens(self,x):
        x = [self.special_tokens[0]] + [torch.cat([i,self.special_tokens[1]]) for i in x]
        return torch.cat(x)
    
def pad_sequence(sequences, batch_first=False, padding_value=0,max_len=None):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences]) if max_len is None else max_len
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor      
          
def collate_creator(pad_token_id,convert_dtype,device,max_len=None):
    # convert_dtype is a dict {'input_ids':x,'attention_mask':atten_mask,'masked_lm_labels':y}
    def collate(listOfTensor):
        x,y = list(zip(*listOfTensor))
        x = pad_sequence(x,batch_first=True,padding_value=pad_token_id,max_len=max_len)
        atten_mask = (x!=pad_token_id).to(torch.float32)
        if y[0] is not None:
            y = torch.tensor(y)
        else:
            y = None
        if convert_dtype is None:
            return {'input_ids':x,'attention_mask':atten_mask,'masked_lm_labels':y}
        else:
            out = {'input_ids':x,'attention_mask':atten_mask,'masked_lm_labels':y}
            out = {k:v.to(device).to(convert_dtype(k)) for k,v in out.items()}
            return out
    return collate
# set config.output_hidden_states=True
class weightedAvg(nn.Module):
    def __init__(self,numLayers,sigma=10):
        self.weight = nn.Parameter(torch.rand(numLayers)/sigma)
    def forward(self,x):
        return torch.matmul(x,nn.functional.softmax(self.weight))

class weightedAvgExtract(nn.Module):
    def __init__(self,extract_element,numLayers=None,sigma=10):
        self.extract_element = extract_element
        if numLayers is None:
            self.weightAvg = None
        else:
            self.weightAvg = weightedAvg(numLayers,sigma)
            
    def forward(self,out):
        if self.weightAvg is None:
            out = out[0][:,:self.extract_element].squeeze()
        else:
            out = torch.stack(out[3],-1) # N,L,d,layers
            out = out[:,:self.extract_element].squeeze()
            out = self.weightAvg(out)
        return out
    
# special_tokens = [torch.tensor([-1,-2], dtype=torch.int16), torch.tensor([-3], dtype=torch.int16)]         
# budget = [0.2,0.3,0.5]
# max_len = 120
# n,l = 10, 15
# q_title = [np.random.randint(0,20,np.random.randint(2,l-5)) for _ in range(n)]
# q_body = [np.random.randint(0,20,np.random.randint(4,l)) for _ in range(n)]
# answer = [np.random.randint(0,20,np.random.randint(4,l)) for _ in range(n)]
# datalist = list(zip(q_title,q_body,answer))
# #datalist = [([1,2],[3,4,5,6],[7,8,9,10,11,12]),]           
# #datalist = [([7,8,9,10,11,12],)]                 
# data_ = DatasetTxts(datalist, max_len, special_tokens, budget, torch.int16)
# train_data = data.DataLoader(data_,batch_size=2,collate_fn=collate_creator(-4,None,None))
# train_data = iter(train_data)
# next(train_data)                
                
                
