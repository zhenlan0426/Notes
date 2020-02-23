#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 10:09:12 2020

@author: will
"""

import torch
import numpy as np
from torch.utils import data
from torch.utils.data import Sampler
#from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_value_
from pytorch_util import trainable_parameter
from transformers import BertForMaskedLM,BertTokenizer

import copy
import time
from random import shuffle
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    UseXLA = True
    device = xm.xla_device()
except:
    UseXLA = False
    device = 'cuda'

try:
    from apex import amp
    UseAmp = True
except:
    UseAmp = False


model_name_mapping = {'bert-large-uncased':[BertForMaskedLM,BertTokenizer]}

class Dataset(data.Dataset):
    def __init__(self, data, max_len, special_tokens, aug=None, token=None, aug_kwargs={}, token_kwargs={'add_special_tokens':False}):
        self.data = data
        self.max_len = max_len - len(special_tokens) # [CLS], [SEP]
        self.aug = aug
        self.token = token
        self.aug_kwargs = aug_kwargs
        self.token_kwargs = token_kwargs
        self.special_tokens = special_tokens
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.aug is None:
            if self.token is None:
                return self.add_special_tokens(self.truncate(self.data[index]))
            else:
                return self.add_special_tokens(self.truncate(self.token.encode(self.data[index],**self.token_kwargs)))
        else:
            return self.add_special_tokens(self.truncate(self.token.encode(self.aug(self.data[index],**self.aug_kwargs),**self.token_kwargs)))
    
    def truncate(self, x):
        if len(x) <= self.max_len:
            return torch.from_numpy(x)
        else:
            start = np.random.randint(0,len(x)-self.max_len)
            return torch.from_numpy(x[start:start+self.max_len])
    
    def add_special_tokens(self,x):
        if len(self.special_tokens)==0:
            return x
        else:
            return torch.cat([torch.tensor([self.special_tokens[0]],dtype=x.dtype)\
                              ,x,torch.tensor([self.special_tokens[1]],dtype=x.dtype)])

class IndexByLength(Sampler):
    # output[self.reverse_index] gives results in original order
    def __init__(self, source_length,limit = 3600):
        ind = np.argsort(source_length)
        sorted_len = np.array(source_length)[ind]
        output_index = []
        temp_index = []
        cum = 0
        for l,i in zip(sorted_len,ind):
            cum += l
            if cum > limit:
                output_index.append(temp_index)
                temp_index = [i]
                cum = l
            else:
                temp_index.append(i)
        if temp_index != []:
            output_index.append(temp_index)
        self.output_index = output_index
        self.reverse_index = np.argsort(ind)
   
    def __iter__(self):
        return iter(self.output_index)
    
    def __len__(self):
        return len(self.output_index)
           
class BySequenceLengthSampler(Sampler):
    '''create batch index by first bucketing sequence to buckets of similar length.To be used as batch_sampler
       data_source is a list of list or a list of np.array
    '''
    def __init__(self, source_length, bucket_boundaries, batch_size):
        # tot_event is a list of [True, False]
        # FalseProb is prob that non-event enters training
        ind_n_len = []
        ind_n_len = [(i,p) for i,p in enumerate(source_length)]
        self.length = len(ind_n_len)
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p,seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]
        for k in data_buckets.keys():
            data_buckets[k] = np.asarray(data_buckets[k])
        self.data_buckets = data_buckets
       
    def __iter__(self):
        iter_list = []
        for k in self.data_buckets.keys():
            np.random.shuffle(self.data_buckets[k])
            iter_list += (np.array_split(self.data_buckets[k]
                           , int(self.data_buckets[k].shape[0]/self.batch_size)))
        shuffle(iter_list) # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.tolist() # as it was stored in an array
   
    def __len__(self):
        return self.length
   
    def element_to_bucket_id(self, x, seq_length):
        boundaries = list(self.bucket_boundaries)
        buckets_min = [np.iinfo(np.int32).min] + boundaries
        buckets_max = boundaries + [np.iinfo(np.int32).max]
        conditions_c = np.logical_and(
          np.less_equal(buckets_min, seq_length),
          np.less(seq_length, buckets_max))
        bucket_id = np.min(np.where(conditions_c))
        return bucket_id

        
def mask_tokens(inputs, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens = tokenizer.build_inputs_with_special_tokens([])

    special_tokens_mask = inputs.eq(special_tokens[0]) | inputs.eq(special_tokens[1]) | labels.eq(tokenizer.pad_token_id)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=inputs.dtype)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

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


def collate_creator(tokenizer,mlm_probability=0.15,max_len=None):
    def collate(listOfTensor):
        listOfTensor = pad_sequence(listOfTensor,batch_first=True,padding_value=tokenizer.pad_token_id,max_len=max_len)
        inputs, labels = mask_tokens(listOfTensor,tokenizer,mlm_probability)
        atten_mask = (listOfTensor!=tokenizer.pad_token_id).to(torch.float32)
        return {'input_ids':inputs,'attention_mask':atten_mask,'masked_lm_labels':labels}
    return collate


def train(opt,model,epochs,train_data,val_data,clip,device,checkFreq=300,scheduler=None,patience=8,accumulation_steps=1):
    since = time.time()
    paras = trainable_parameter(model)
    counter = 0
    lossBest = 1e6
    bestWeight = None
       
    opt.zero_grad()
    for epoch in range(epochs):
        # training #
        model.train()
        train_loss = 0
        for i,data_kwarg in enumerate(train_data):
            data_kwarg = {k:v.to(device).to(torch.int64) if v.dtype in (torch.int16,torch.int32) else v.to(device) for k,v in data_kwarg.items()}
            loss = model(**data_kwarg)[0]/accumulation_steps
            train_loss = train_loss + loss.item()
            if UseAmp:
                with amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
                clip_grad_value_(amp.master_params(opt),clip)
            else:
                loss.backward()
                clip_grad_value_(paras,clip)
                
            if (i+1) % accumulation_steps == 0:
                if UseXLA:
                    xm.optimizer_step(opt)
                else:
                    opt.step()
                opt.zero_grad()
            
            if (i+1) % checkFreq == 0:
                train_loss = train_loss/checkFreq
                val_loss = 0
                model.eval()
                with torch.no_grad():
                    for data_kwarg in val_data:
                        data_kwarg = {k:v.to(device).to(torch.int64) if v.dtype in (torch.int16,torch.int32) else v.to(device) for k,v in data_kwarg.items()}
                        loss = model(**data_kwarg)[0]/accumulation_steps
                        val_loss = val_loss + loss.item()
                val_loss = val_loss/len(val_data)
                print('epoch:{}, batch:{}, train_loss:{}, val_loss:{}\n'.format(epoch,i,train_loss,val_loss))

                # save model & early stop
                if val_loss < lossBest:
                    lossBest = val_loss
                    bestWeight = copy.deepcopy(model.state_dict())
                    counter = 0
                else:
                    counter += 1
                    if counter >= patience:
                        print('----early stop at epoch {}, batch {}----'.format(epoch,i))
                        time_elapsed = time.time() - since
                        print('Training completed in {} mins'.format(time_elapsed/60))
                        return model,bestWeight,lossBest
                if scheduler is not None:
                    scheduler.step(val_loss)
                train_loss = 0
                val_loss = 0
                model.train()
            
    time_elapsed = time.time() - since
    print('Training completed in {} mins'.format(time_elapsed/60))
    return model,bestWeight,lossBest




