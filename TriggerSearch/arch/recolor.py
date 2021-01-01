import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class new(nn.Module):
    def __init__(self,nlayers,nh,sz=1):
        super(new,self).__init__()
        self.layers=nn.ModuleList();
        if nlayers==1:
            self.layers.append(nn.Conv2d(3,3,2*sz-1,padding=sz-1));
        else:
            self.layers.append(nn.Conv2d(3,nh,2*sz-1,padding=sz-1));
            for i in range(nlayers-2):
                self.layers.append(nn.Conv2d(nh,nh,2*sz-1,padding=sz-1));
            
            self.layers.append(nn.Conv2d(nh,3,2*sz-1,padding=sz-1));
        
        return;
    
    def forward(self,I):
        h=I;
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
        
        h=self.layers[-1](h);
        return h
    