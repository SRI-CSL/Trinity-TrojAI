import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image
import torchvision.datasets.folder
import torchvision.transforms.functional as Ft
import torchvision.transforms as Ts
import PIL.Image as Image

import torch.utils.data.dataloader

import os
import time
import math
import numpy
from itertools import tee

import util.db as db



#Inheriting the general DB-based dataloader class to save a few lines
#db.Dataloader provides
#.data: the DB
#.load(): load DB from disk
#.cuda(): send DB to cuda
#.cpu(): send DB to cpu
class new(db.Dataloader):
    #Define constants here
    
    #Total number of annotations
    def size(self):
        return len(self.data['table_ann']);
    
    def device(self):
        return self.data['table_ann']['flip_loss'].device;
    
    #Define a preprocessing procedure.
    #Often something I forgot to do with data
    def preprocess(self):
        #self.data['table_ann']['flip_x']=self.data['table_ann']['flip_x']-0.5;
        #self.data['table_ann']['flip_y']=self.data['table_ann']['flip_y']-0.5;
        
        stuff={};
        for k in ['attrib_logits','filter_log']:
            if k in self.data['table_ann'].fields():
                self.data['table_ann'][k]=self.data['table_ann'][k].data;
                stuff[k]={'mean':self.data['table_ann'][k].mean(0,keepdim=True),'std':self.data['table_ann'][k].std(0,keepdim=True)};
                self.data['table_ann'][k]-=stuff[k]['mean'];
                self.data['table_ann'][k]/=stuff[k]['std']+1e-5;
                self.data['table_ann'][k]=self.data['table_ann'][k].float()
        
        return stuff; 
    
    def generate_random_crossval_split(self,pct=0.8,aug='',seed=None):
        if not(seed is None):
            rng_state=torch.random.get_rng_state();
            torch.random.manual_seed(seed);
        
        ind=torch.randperm(self.size()).long();
        if not(seed is None):
            torch.random.set_rng_state(rng_state);
        
        ntrain=math.ceil(pct*self.size());
        ind_train=ind[:ntrain];
        ind_test=ind[ntrain:];
        
        split_train={'index':ind_train};
        split_test={'index':ind_test};
        
        data_split_train=self.subsample(split_train);
        data_split_test=self.subsample(split_test);
        return data_split_train,data_split_test;
    
    
    #Sample a subset from the dataset
    def subsample(self,split):
        ind=split['index'];
        table_ann=self.data['table_ann'].select_by_index(ind.tolist());
        d=db.DB({'table_ann':table_ann});
        data=type(self)(d);
        return data;
    
    #Training iterator
    def batches(self,batch_size=256,seed=None,shuffle=False):
        if not seed is None:
            rng_state=torch.random.get_rng_state();
            torch.random.manual_seed(seed);
        
        #Shuffle dataset here using pytorch seed
        if shuffle:
            ind=torch.randperm(len(self.data['table_ann']));
            table_ann=self.data['table_ann'].select_by_index(ind);
        else:
            table_ann=self.data['table_ann'];
        
        if not seed is None:
            torch.random.set_rng_state(rng_state);
        
        #Loops through the dataset
        n=self.size();
        for i in range(0,n,batch_size):
            r=min(i+batch_size,n);
            table_batch=db.Table(table_ann[i:r]);
            table_batch['label']=torch.LongTensor(table_batch['label']);
            yield table_batch;
        
        return
    