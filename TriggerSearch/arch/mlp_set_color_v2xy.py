import torch
import torchvision.models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
import math
from collections import OrderedDict as OrderedDict
import copy

class MLP(nn.Module):
    def __init__(self,ninput,nh,noutput,nlayers):
        super().__init__()
        
        self.layers=nn.ModuleList();
        self.bn=nn.LayerNorm(ninput);
        if nlayers==1:
            self.layers.append(nn.Linear(ninput,noutput));
        else:
            self.layers.append(nn.Linear(ninput,nh));
            for i in range(nlayers-2):
                self.layers.append(nn.Linear(nh,nh));
            
            self.layers.append(nn.Linear(nh,noutput));
        
        self.ninput=ninput;
        self.noutput=noutput;
        return;
    
    def forward(self,x):
        h=x.view(-1,self.ninput);
        #h=self.bn(h);
        for i in range(len(self.layers)-1):
            h=self.layers[i](h);
            h=F.relu(h);
            #h=F.dropout(h,training=self.training);
        
        h=self.layers[-1](h);
        return h

#Input n x K x K x ninput
#Output n x K x 2nh
class same_diff_encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fv):
        rounds=fv.shape[0];
        nclasses=fv.shape[1];
        assert fv.shape[2]==nclasses;
        
        h=fv.view(rounds*nclasses*nclasses,-1);
        h=self.encoder(h);
        h=h.view(rounds,nclasses*nclasses,-1);
        
        ind_diag=list(range(0,nclasses*nclasses,nclasses+1));
        ind_off_diag=list(set(list(range(nclasses*nclasses))).difference(set(ind_diag)))
        ind_diag=torch.LongTensor(list(ind_diag)).to(h.device)
        ind_off_diag=torch.LongTensor(list(ind_off_diag)).to(h.device)
        h_diag=h[:,ind_diag,:];
        h_off_diag=h[:,ind_off_diag,:].contiguous().view(rounds,nclasses,nclasses-1,-1).mean(2);
        return torch.cat((h_diag,h_off_diag),dim=2);

#Input n x K x ninput
#Output n x nh
class encoder(nn.Module):
    def __init__(self,ninput,nh,nlayers):
        super().__init__()
        self.encoder=MLP(ninput,nh,nh,nlayers);
    
    def forward(self,fv):
        rounds=fv.shape[0];
        nclasses=fv.shape[1];
        
        h=fv.view(rounds*nclasses,-1);
        h=self.encoder(h);
        h=h.view(rounds,nclasses,-1).mean(1);
        return h;


class new(nn.Module):
    def __init__(self,params):
        super(new,self).__init__()
        nclasses_=5;
        nim_=100;
        ninput=nclasses_*nclasses_*6;
        nh=params.nh;
        nh2=params.nh2;
        nlayers=params.nlayers
        nlayers2=params.nlayers2
        nlayers3=params.nlayers3
        
        nrounds=8;
        nchkpt=1;
        nchkpt2=6;
        self.encoder_fv_color=encoder(nchkpt*27 ,nh2,nlayers);
        self.encoder_fv_color_xy=encoder(nchkpt*51 ,nh2,nlayers);
        
        nrounds=2;
        nchkpt=11;
        nchkpt2=6;
        #self.encoder_effect_blob=same_diff_encoder(nrounds*nchkpt2*2,nh2,nlayers3);
        self.encoder_fv_blob=encoder(nrounds*nchkpt*8 ,nh,nlayers3);
        
        
        self.encoder_combined=MLP(nh+nh2*2,nh+nh2*2,2,nlayers2); #+46
        self.w=nn.Parameter(torch.Tensor(1).fill_(1));
        self.b=nn.Parameter(torch.Tensor(1).fill_(0));
        
        return;
    
    def forward(self,data_batch):
        #Deal with the case of a single item
        if not isinstance(data_batch['color_loss'],list):
            data_batch_exp={};
            for k in ['flip_loss','flip_area','flip_mass','flip_div','flip_blob_x','flip_blob_y','flip_blob_w','flip_blob_h']:
                data_batch_exp[k]=[data_batch[k]];
            
            for k in ['color_loss','color_loss2','color_brick','color_div']:
                data_batch_exp[k]=[data_batch[k]];
            
            for k in ['color_loss_xy','color_loss2_xy','color_brick_xy','color_div_xy']:
                data_batch_exp[k]=[data_batch[k]];
            
            data_batch=data_batch_exp;
        
        color_loss=data_batch['color_loss'];
        b=len(color_loss);
        
        h=[];
        #Have to process one by one due to variable nim & nclasses
        for i in range(b):
            #Then encode the nclasses features
            color_loss=data_batch['color_loss'][i].cuda();
            nclasses=color_loss.shape[0];
            nrounds=color_loss.shape[1];
            
            color_loss2=data_batch['color_loss2'][i].cuda();
            color_brick=(data_batch['color_brick'][i].cuda()-0.5).view(nclasses,nrounds,-1);
            color_div=data_batch['color_div'][i].cuda();
            
            fv_color=torch.cat((color_loss,color_loss2,color_brick,color_div),dim=2)
            fv_color=fv_color.view(1,nclasses*nrounds,1*27);
            fv_color=self.encoder_fv_color(fv_color).mean(0);
            
            #Then encode the nclasses features
            color_loss=data_batch['color_loss_xy'][i].cuda();
            nclasses=color_loss.shape[0];
            nrounds=color_loss.shape[1];
            
            color_loss2=data_batch['color_loss2_xy'][i].cuda();
            color_brick=(data_batch['color_brick_xy'][i].cuda()-0.5).view(nclasses,nrounds,-1);
            color_div=data_batch['color_div_xy'][i].cuda();
            
            fv_color_xy=torch.cat((color_loss,color_loss2,color_brick,color_div),dim=2)
            fv_color_xy=fv_color_xy.view(1,nclasses*nrounds,1*51);
            fv_color_xy=self.encoder_fv_color_xy(fv_color_xy).mean(0);
            
            
            #Then encode the nclasses features
            flip_loss=data_batch['flip_loss'][i].cuda();
            flip_area=data_batch['flip_area'][i].cuda()*5-1;
            flip_mass=data_batch['flip_mass'][i].cuda()*5-1;
            flip_div=data_batch['flip_div'][i].cuda()*5;
            nclasses=flip_loss.shape[0];
            
            flip_x=data_batch['flip_blob_x'][i].cuda();
            flip_y=data_batch['flip_blob_y'][i].cuda();
            flip_w=data_batch['flip_blob_w'][i].cuda()*5;
            flip_h=data_batch['flip_blob_h'][i].cuda()*5;
            fv_trigger=torch.stack((flip_loss,flip_area,flip_mass,flip_div,flip_x,flip_y,flip_w,flip_h),dim=2)[:,:,:,:].contiguous()
            fv_trigger=fv_trigger.view(1,nclasses,22*8);
            fv_trigger=self.encoder_fv_blob(fv_trigger).mean(0);
            
            fv=torch.cat((fv_color,fv_color_xy,fv_trigger),dim=0);
            h.append(fv);
        
        h=torch.stack(h,dim=0);
        #h=torch.cat((h,attrib_logits,attrib_histogram,filter_log,n),dim=1);
        #print(h.shape);
        h=self.encoder_combined(h);
        h=torch.tanh(h)*8;
        return h
    
    def logp(self,data_batch):
        h=self.forward(data_batch);
        return h[:,1]-h[:,0];
    