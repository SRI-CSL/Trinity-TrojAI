#Python2,3 compatible headers
from __future__ import unicode_literals,division
from builtins import int
from builtins import range

#System packages
import torch
from torch.autograd import Variable,grad
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy
import scipy
import scipy.misc
import math
import time
import random
import argparse
import sys
import os
import re
import copy
import importlib
from collections import namedtuple
from collections import OrderedDict
from itertools import chain

import PIL.Image
import torchvision.datasets.folder
import torchvision.transforms.functional as Ft
import torchvision.transforms as Ts
import PIL.Image as Image

import torch.utils.data.dataloader as myDataLoader
import skimage.io





def loadim(fname):
    img = skimage.io.imread(fname)
    img = img.astype(dtype=numpy.float32)
    h, w, c = img.shape
    dx = int((w - 224) / 2)
    dy = int((w - 224) / 2)
    img = img[dy:dy+224, dx:dx+224, :]
    # perform tensor formatting and normalization explicitly
    # convert to CHW dimension ordering
    img = numpy.transpose(img, (2, 0, 1))
    # convert to NCHW dimension ordering
    img = numpy.expand_dims(img, 0)
    # normalize the image
    img = img / 255.0
    batch_data = torch.from_numpy(img).cuda();
    return batch_data;


def extract_fv(model_filepath,examples_dirpath,example_img_format='png',model_id=0):
    #Hyper params
    delta=10;
    nsteps=51;
    lr=3e-2;
    wenergy=10;
    warea=10;
    wblob=10;
    wdiv=2;
    batch=5;
    rounds=2;
    multi=30;
    checkpoints=list(range(0,nsteps,5));
    checkpoints2=list(range(0,nsteps,10));
    
    def overlay(im,a,x,aug=False):
        N=im.shape[0];
        
        x_=torch.sigmoid(x*multi);
        a_=torch.sigmoid(a*multi);
        
        if aug:
            #Randomly perturb the trigger using affine transforms so we find robust triggers and not adversarial noise
            transform=[];
            for _ in range(N):
                #scale
                sz=float(torch.Tensor(1).uniform_(0.8,1.2));
                #smol rotation
                theta=float(torch.Tensor(1).uniform_(-3.14/6,3.14/6));
                #5% imsz offset
                pad=0.3;
                offset=(torch.Tensor(2).uniform_(-pad,pad)).tolist();
                transform.append(torch.Tensor([[sz*math.cos(theta),-sz*math.sin(theta),offset[0]],[sz*math.sin(theta),sz*math.cos(theta),offset[1]]]));
            
            transform=torch.stack(transform,dim=0)
            grid=F.affine_grid(transform,im.size()).cuda();
            #Synthesize trigger
            a_=F.grid_sample(a_.repeat(N,1,1,1),grid);
            x_=F.grid_sample(x_.repeat(N,1,1,1),grid);
        
        im_edit=(1-a_)*im+a_*x_;
        
        return im_edit;
    
    def get_energy(a):
        a_=torch.sigmoid(a*multi);
        total=a_.sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True)+1e-8;
        energy=torch.sqrt(total/a.shape[2]/a.shape[3]+1e-12);
        return energy.mean();
    
    def get_area(a):
        total=torch.sigmoid(a*multi*2+6).sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True)+1e-8;
        area=torch.sqrt(total/a.shape[2]/a.shape[3]+1e-12);
        return area.mean();
    
    def get_blob(a):
        w=a.shape[3];
        h=a.shape[2];
        a_=torch.sigmoid(a*multi);
        total=a_.sum(3,keepdim=True).sum(2,keepdim=True).sum(1,keepdim=True)+1e-8;
        
        #3. Blob size of the edit
        wt=torch.Tensor(list(range(w)))/w-0.5;
        wt=wt.view(1,1,1,w).cuda();
        ht=torch.Tensor(list(range(h)))/h-0.5;
        ht=ht.view(1,1,h,1).cuda();
        
        opacity_dist=a_/total;
        Ew=(opacity_dist*wt).sum(3).sum(2).sum(1);
        Eh=(opacity_dist*ht).sum(3).sum(2).sum(1);
        Ew2=(opacity_dist*(wt**2)).sum(3).sum(2).sum(1);
        Eh2=(opacity_dist*(ht**2)).sum(3).sum(2).sum(1);
        varw=Ew2-Ew**2;
        varh=Eh2-Eh**2;
        stdw=torch.sqrt(varw.clamp(min=0)+1e-12);
        stdh=torch.sqrt(varh.clamp(min=0)+1e-12);
        
        return Ew.mean(),Eh.mean(),stdw.mean(),stdh.mean();
    
    
    def loss_overlay(energy,area,blob,progress):
        x,y,w,h=blob;
        loss=wenergy*progress*energy+warea*progress*area+wblob*progress*(w+h); #*progress
        return loss;
    
    
    
    model=torch.load(model_filepath);
    imnames=[fname for fname in os.listdir(examples_dirpath) if fname.endswith(example_img_format)];
    model=model.cuda()
    model.eval();
    
    #First run through the images and pick 1 per class
    nim=len(imnames);
    im_by_class={};
    nclasses=-1;
    for i in range(0,nim):
        imname=imnames[i];
        im=loadim(os.path.join(examples_dirpath,imname)).cuda();
        scores=model(im.cuda()).data.cpu();
        nclasses=scores.shape[1];
        _,pred=scores.max(dim=1);
        pred=int(pred);
        if not pred in im_by_class:
            im_by_class[pred]=[];
        
        im_by_class[pred].append(imname);
    
    imnames=[];
    for c in range(nclasses):
        if c in im_by_class:
            imnames.append(im_by_class[c]);
        else:
            imnames.append([])
    
    #Load images, max 5 per class
    ims=[];
    for c in range(nclasses):
        if len(imnames[c])>0:
            #Load max 5 images
            N=min(batch,len(imnames[c]))
            ims_c=[];
            for i in range(N):
                imname=imnames[c][i];
                ims_c.append(loadim(os.path.join(examples_dirpath,imname)).cuda())
            
            ims_c=torch.cat(ims_c,dim=0);
            ims.append(ims_c);
        else:
            print('class %d missing ims'%c)
            ims.append(None);
    
    
    #Collect the following features
    nim=len(imnames);
    nchkpt=len(checkpoints);
    nchkpt2=len(checkpoints2);
    scores_pred=torch.zeros(nclasses,rounds,nclasses);
    flip_loss=torch.zeros(nclasses,rounds,nchkpt);
    flip_area=torch.zeros(nclasses,rounds,nchkpt);
    flip_mass=torch.zeros(nclasses,rounds,nchkpt);
    flip_div=torch.zeros(nclasses,rounds,nchkpt);
    flip_blob_x=torch.zeros(nclasses,rounds,nchkpt);
    flip_blob_y=torch.zeros(nclasses,rounds,nchkpt);
    flip_blob_w=torch.zeros(nclasses,rounds,nchkpt);
    flip_blob_h=torch.zeros(nclasses,rounds,nchkpt);
    flip_blob_h=torch.zeros(nclasses,rounds,nchkpt);
    flip_loss_before=torch.zeros(nclasses,nclasses,rounds,nchkpt2);
    flip_loss_after=torch.zeros(nclasses,nclasses,rounds,nchkpt2);
    
    #Loop through the images in batches
    t0=time.time();
    for c in range(nclasses):
        if not (ims[c] is None):
            #Prediction
            scores=F.log_softmax(model(ims[c]),dim=1).data.cpu();
            scores_pred[c,:]=scores.mean(0);
            
            #Perform image editing -  any target  -  multiple rounds
            triggers=[];
            for round in range(rounds):
                print(c,round)
                triggers_round=[]
                
                w=ims[c].shape[3];
                h=ims[c].shape[2];
                a=torch.Tensor(1,1,h,w).uniform_(-0.01,0.01)-0.1; #alpha: opacity. Range 0-1, control center of mass & magnitude
                content=torch.Tensor(1,3,h,w).uniform_(-0.01,0.01); #x: Content. Range 0-1
                a=a.cuda().requires_grad_();
                content=content.cuda().requires_grad_();
                opt=optim.Adam([a,content],lr=lr,betas=(0.5,0.7));
                
                #Gradient descend
                for k in range(nsteps+delta):
                    #Randomly perturb the trigger using affine transforms so we find robust triggers and not adversarial noise
                    im_edit=overlay(ims[c],a,content,aug=True);
                    
                    #1. Compute loss to target
                    scores=F.softmax(model(im_edit),dim=1);
                    loss_target=scores[:,c].mean();
                    
                    #2. Compute prior term on overlay
                    progress=max(0,(k-delta)/nsteps);
                    energy=get_energy(a);
                    area=get_area(a);
                    blob=get_blob(a);
                    loss_a=loss_overlay(energy,area,blob,progress);
                    
                    #3. Add diversity
                    loss_div=0;
                    trigger=F.normalize((torch.sigmoid(content*multi)*torch.sigmoid(a*multi)).view(-1),dim=0);
                    for prev_trigger in triggers:
                        loss_div=loss_div+(trigger*prev_trigger).sum();
                    
                    loss_div=loss_div*wdiv;
                    
                    loss=loss_target+loss_a+loss_div;
                    #print('%d-%d, %.4f: target %.4f, energy %.4f, blob %.4f, div %.4f, time %.2f'%(c,k,loss,loss_target,energy,(blob[2]+blob[3])/2,loss_div,time.time()-t0));
                    loss.backward();
                    opt.step();
                    opt.zero_grad();
                    
                    if k-delta in checkpoints:
                        #Extract features
                        #Old stuff
                        chkpt_id=checkpoints.index(k-delta);
                        x,y,w,h=blob;
                        flip_loss[c,round,chkpt_id]=float(loss_target);
                        flip_mass[c,round,chkpt_id]=float(energy);
                        flip_area[c,round,chkpt_id]=float(area);
                        flip_div[c,round,chkpt_id]=float(loss_div);
                        flip_blob_x[c,round,chkpt_id]=float(x);
                        flip_blob_y[c,round,chkpt_id]=float(y);
                        flip_blob_w[c,round,chkpt_id]=float(w);
                        flip_blob_h[c,round,chkpt_id]=float(h);
                    
                    #New stuff:Paste triggers onto different classes and see if prediction changes
                    if k-delta in checkpoints2:
                        chkpt_id=checkpoints2.index(k-delta);
                    
                    
                    if k-delta in checkpoints:
                        #Record solution for diversity 
                        with torch.no_grad():
                            trigger=F.normalize((torch.sigmoid(content.data*multi)*torch.sigmoid(a.data*multi)).view(-1),dim=0);
                            triggers_round.append(trigger);
                        
                        #Visualize result
                        if False:
                            #Save image checkpoint for debugging
                            for i in range(N):
                                im=im_edit[i].data.cpu();
                                try:
                                    os.mkdir('temp');
                                except:
                                    pass;
                                try:
                                    os.mkdir('temp/id-%08d'%(model_id));
                                except:
                                    pass;
                                
                                fname='temp/id-%08d/%s_round%03d_iter%03d.png'%(model_id,imnames[c][i],round,k-delta);
                                torchvision.utils.save_image(im,fname);
                
                
                triggers=triggers+triggers_round;
    
    
    
    fvs={'scores_pred':scores_pred,'flip_loss':flip_loss,'flip_mass':flip_mass,'flip_area':flip_area,'flip_div':flip_div,'flip_blob_x':flip_blob_x,'flip_blob_y':flip_blob_y,'flip_blob_w':flip_blob_w,'flip_blob_h':flip_blob_h};
    return fvs;

#Task: try to identify trojan triggers from images
if __name__ == "__main__":
    import util.smartparse as smartparse
    # Training settings
    def default_params():
        params=smartparse.obj();
        params.start=-1
        params.end=-1
        return params
    
    
    params = smartparse.parse()
    params = smartparse.merge(params, default_params())
    params.argv=sys.argv;
    
    
    root='data/round3-dataset-train/models';
    models=os.listdir(root);
    models=sorted(models);
    
    t0=time.time();
    fvs=[];
    ids=[];
    torch.manual_seed(0)
    
    id_list=torch.randperm(1008).tolist();
    if params.start>=0:
        if params.end>=0:
            id_list=id_list[params.start:params.end];
        else:
            id_list=id_list[params.start:];
    
    for id in id_list:
        fname=os.path.join(root,models[id],'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        model_filepath=os.path.join(root,models[id],'model.pt');
        examples_dirpath=os.path.join(root,models[id],'clean_example_data');
        data=extract_fv(model_filepath,examples_dirpath,model_id=id);
        data['label']=label;
        fvs.append(data);
        ids.append(models[id]);
        print('%d, time %.3f'%(id,time.time()-t0));
        if params.start>=0:
            torch.save({'fvs':fvs,'ids':ids},'fvs_polygon_r3_v2_%d_%d.pt'%(params.start,params.end));
        else:
            torch.save({'fvs':fvs,'ids':ids},'fvs_polygon_r3_v2.pt');

