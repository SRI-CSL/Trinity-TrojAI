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
    # convert image to a gpu tensor
    img = img / 255.0
    batch_data = torch.from_numpy(img).cuda();
    return batch_data;


def extract_fv(model_filepath,examples_dirpath,example_img_format='png',model_id=0):
    #Hyper params
    nsteps=11;
    nrounds=8;
    lr=3e-2;
    wl2=1e-4;
    wdiv=0.05;
    batch=5;
    nlayers=2;
    nh=16;
    
    checkpoints=[10];
    checkpoints2=[10];
    
    def recolor(im,recolor_net,aug=False):
        N=im.shape[0];
        if aug:
            im_=[];
            for i in range(N):
                im_i=im[i];
                
                
                brightness_factor=float(torch.Tensor(1).uniform_(0.8,1.2));
                contrast_factor=float(torch.Tensor(1).uniform_(0.8,1.2));
                gamma=float(torch.Tensor(1).uniform_(0.8,1.2));
                hue_factor=float(torch.Tensor(1).uniform_(-0.1,0.1));
                
                im_i=Ft.adjust_brightness(im_i,brightness_factor);
                im_i=Ft.adjust_contrast(im_i,contrast_factor);
                im_i=Ft.to_tensor(Ft.adjust_gamma(Ft.to_pil_image(im_i.cpu()),gamma)).cuda();
                im_i=Ft.to_tensor(Ft.adjust_hue(Ft.to_pil_image(im_i.cpu()),hue_factor)).cuda();
                
                im_.append(im_i);
            
            im=torch.stack(im_,dim=0);
        
        
        imout=recolor_net(im-0.5)+im-0.5;
        imout=F.sigmoid(imout*5);
        return imout;
    
    def get_l2(recolor_net):
        l2=0;
        for param in recolor_net.parameters():
            l2=l2+(param**2).sum();
        
        return l2;
    
    def get_color_brick(recolor_net):
        return recolor_net.characterize();
    
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
    scores_pred=torch.zeros(nclasses,nclasses);
    color_loss=torch.zeros(nclasses,nrounds,nchkpt);
    color_loss2=torch.zeros(nclasses,nrounds,nchkpt);
    color_brick=torch.zeros(nclasses,nrounds,nchkpt,48);
    color_l2=torch.zeros(nclasses,nrounds,nchkpt);
    color_div=torch.zeros(nclasses,nrounds,nchkpt);
    
    color_loss_before=torch.zeros(nclasses,nclasses,nrounds,nchkpt2);
    color_loss_after=torch.zeros(nclasses,nclasses,nrounds,nchkpt2);
    
    import arch.recolor_xy as arch_recolor
    #Loop through the images in batches
    t0=time.time();
    for c in range(nclasses):
        if not (ims[c] is None):
            #Prediction
            scores=F.log_softmax(model(ims[c]),dim=1).data.cpu();
            scores_pred[c,:]=scores.mean(0);
            
            color_bricks=[];
            for round_id in range(nrounds):
                recolor_net=arch_recolor.new(nlayers,nh).cuda();
                opt=optim.Adam(recolor_net.parameters(),lr=lr,betas=(0.5,0.7));
                
                #Gradient descend
                color_bricks_r=[];
                for k in range(nsteps):
                    #Randomly perturb the trigger using affine transforms so we find robust triggers and not adversarial noise
                    im_edit=recolor(ims[c],recolor_net,aug=True);
                    
                    #1. Compute loss to some target
                    scores=F.softmax(model(im_edit),dim=1);
                    loss_target=scores[:,c].mean();
                    
                    scoresv2=F.log_softmax(model(im_edit),dim=1);
                    scoresv2=torch.cat((scoresv2[:,:c],scoresv2[:,c+1:]),dim=1);
                    loss_target2=-torch.logsumexp(scoresv2*(1+k/nsteps*2),dim=1).mean()/3;
                    
                    #2. Compute prior term on overlay
                    l2=get_l2(recolor_net);
                    
                    #3. Compute divergence
                    color_brick_i=get_color_brick(recolor_net);
                    loss_div_i=0
                    for cb in color_bricks:
                        loss_div_i=loss_div_i-((color_brick_i-cb)**2).mean();
                    
                    
                    loss=loss_target2+wl2*l2+wdiv*loss_div_i;
                    loss.backward();
                    opt.step();
                    opt.zero_grad();
                    
                    if k in checkpoints:
                        #print('%d-%d, %.4f: target %.4f-%.4f, l2 %.4f, div %.4f, time %.2f'%(c,k,loss,loss_target,loss_target2,l2,loss_div_i,time.time()-t0));
                        #Extract features
                        #Old stuff
                        chkpt_id=checkpoints.index(k);
                        color_loss[c,round_id,chkpt_id]=float(loss_target);
                        color_loss2[c,round_id,chkpt_id]=float(loss_target2);
                        color_l2[c,round_id,chkpt_id]=float(l2);
                        color_div[c,round_id,chkpt_id]=float(loss_div_i);
                        color_brick[c,round_id,chkpt_id,:]=get_color_brick(recolor_net).cpu();
                    
                    #New stuff:Paste triggers onto different classes and see if prediction changes
                    if k in checkpoints2:
                        chkpt_id=checkpoints2.index(k);
                        color_bricks_r.append(color_brick_i.data);
                    
                    
                    if k in checkpoints:
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
                                
                                fname='temp/id-%08d/%s_round%d_iter%03d.png'%(model_id,imnames[c][i],round_id,k);
                                torchvision.utils.save_image(im,fname);
                
                color_bricks=color_bricks+color_bricks_r;
    
    
    
    fvs={'scores_pred':scores_pred,'color_loss':color_loss,'color_loss2':color_loss2,'color_brick':color_brick,'color_l2':color_l2,'color_div':color_div};
    return fvs;

#Task: try to identify trojan triggers from images
if __name__ == "__main__":
    # Training settings
    import util.smartparse as smartparse
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
            id_list=id_list[params.start:params.end+1];
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
        print('%d, time %.3f'%(len(ids),time.time()-t0));
        if params.start>=0:
            torch.save({'fvs':fvs,'ids':ids},'fvs_color_r3_v2xy_%d_%d.pt'%(params.start,params.end));
        else:
            torch.save({'fvs':fvs,'ids':ids},'fvs_color_r3_v2xy.pt');
    






