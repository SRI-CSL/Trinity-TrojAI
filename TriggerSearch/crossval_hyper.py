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

import util.db as db
import util.smartparse as smartparse
import util.file
import util.session_manager as session_manager
import dataloader

import sklearn.metrics
from hyperopt import hp, tpe, fmin

# Training settings
def default_params():
    params=smartparse.obj();
    #Data
    params.nsplits=8;
    params.pct=0.8
    #Model
    params.arch='arch.mlpv2';
    params.nh=256;
    params.nlayers=3;
    
    #Optimization
    params.batch=256;
    params.lr=1e-3;
    params.epochs=300;
    params.decay=1e-4;
    
    
    #MISC
    params.session_dir=None;
    params.budget=10000;
    return params

def create_session(params):
    session=session_manager.Session(session_dir=params.session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    util.file.write_json(session.file('params.json'),pmvs); #Write a human-readable parameter json
    session.file('model','dummy');
    return session;


params = smartparse.parse()
params = smartparse.merge(params, default_params())
params.argv=sys.argv;

data=dataloader.new('data_r3v3.pt');
data.cuda();
params.stuff=data.preprocess();


session=create_session(params);
params.session=session;

#Hyperparam search config
hp_config=[];

#   Architectures
#archs=['arch.mlpv2','arch.mlpv3','arch.mlpv4','arch.mlpv5','arch.mlpv6'];
archs=['arch.mlp_set_color_v2xy'];

hp_config.append(hp.choice('arch',archs));
hp_config.append(hp.qloguniform('nh',low=math.log(16),high=math.log(384),q=4));
hp_config.append(hp.qloguniform('nh2',low=math.log(16),high=math.log(384),q=4));
hp_config.append(hp.quniform('nlayers',low=1,high=10,q=1));
hp_config.append(hp.quniform('nlayers2',low=1,high=10,q=1));
hp_config.append(hp.quniform('nlayers3',low=1,high=10,q=1));
#   OPT
hp_config.append(hp.qloguniform('epochs',low=math.log(5),high=math.log(200),q=1));
hp_config.append(hp.loguniform('lr',low=math.log(1e-4),high=math.log(5e-2)));
hp_config.append(hp.loguniform('decay',low=math.log(1e-6),high=math.log(1e-3)));
hp_config.append(hp.qloguniform('batch',low=math.log(16),high=math.log(64),q=1));

#Function to compute performance
def configure_pipeline(params,arch,nh,nh2,nlayers,nlayers2,nlayers3,epochs,lr,decay,batch):
    params_=smartparse.obj();
    params_.arch=arch;
    params_.nh=int(nh);
    params_.nh2=int(nh2);
    params_.nlayers=int(nlayers);
    params_.nlayers2=int(nlayers2);
    params_.nlayers3=int(nlayers3);
    params_.epochs=epochs;
    params_.lr=lr;
    params_.batch=batch;
    params_=smartparse.merge(params_,params);
    return params_;

crossval_splits=[];
for i in range(params.nsplits):
    data_train,data_test=data.generate_random_crossval_split(pct=params.pct);
    data_train,data_val=data_train.generate_random_crossval_split();
    crossval_splits.append((data_train,data_val,data_test));

best_loss_so_far=-1e10;
def run_crossval(p):
    global best_loss_so_far
    max_batch=16;
    arch,nh,nh2,nlayers,nlayers2,nlayers3,epochs,lr,decay,batch=p;
    params_=configure_pipeline(params,arch,nh,nh2,nlayers,nlayers2,nlayers3,epochs,lr,decay,batch);
    arch_=importlib.import_module(params_.arch);
    #Random splits N times
    auc=[];
    ce=[];
    cepre=[];
    results_by_key={};
    t0=time.time();
    ensemble=[];
    for split_id,split in enumerate(crossval_splits):
        data_train,data_val,data_test=split;
        net=arch_.new(params_).cuda();
        opt=optim.Adam(net.parameters(),lr=params_.lr); #params_.lr
        
        #Train loop
        best_loss=-1e10;
        best_net=copy.deepcopy(net);
        
        for iter in range(params_.epochs):
            net.train();
            loss_total=[];
            for data_batch in data_train.batches(params_.batch,shuffle=True):
                opt.zero_grad();
                net.zero_grad();
                data_batch.cuda();
                C=data_batch['label'];
                data_batch.delete_column('label');
                scores_i=net(data_batch);
                
                #loss=F.binary_cross_entropy_with_logits(scores_i,C.float());
                spos=scores_i.gather(1,C.view(-1,1)).mean();
                sneg=torch.exp(scores_i).mean();
                loss=-(spos-sneg+1);
                #print(float(loss))
                l2=0;
                for p in net.parameters():
                    l2=l2+(p**2).sum();
                
                loss=loss+l2*params_.decay;
                loss.backward();
                loss_total.append(float(loss));
                opt.step();
            
            loss_total=sum(loss_total)/len(loss_total);
            
            #Eval every epoch 
            #net.eval();
            #scores=[];
            #gt=[]
            #for data_batch in data_val.batches(max_batch):
            #    data_batch.cuda();
            #    
            #    C=data_batch['label'];
            #    data_batch.delete_column('label');
            #    scores_i=net(data_batch);
            #    scores.append(scores_i.data.cpu());
            #    gt.append(C.data.cpu());
            
            #scores=torch.cat(scores,dim=0);
            #gt=torch.cat(gt,dim=0);
            
            #auc_i=sklearn.metrics.roc_auc_score(gt.numpy(),scores.numpy());
            #loss_i=float(F.binary_cross_entropy_with_logits(scores,gt.float()));
            #if best_loss<auc_i:
            #    best_loss=auc_i;
            #    best_net=copy.deepcopy(net);
            
            #print('train %.4f, loss %.4f, auc %.4f'%(float(loss_total),float(loss_i),float(auc_i)))
        
        #Temperature-scaling calibration on val
        #net=best_net;
        net.eval();
        scores=[];
        gt=[];
        for data_batch in data_val.batches(max_batch):
            data_batch.cuda();
            
            C=data_batch['label'];
            data_batch.delete_column('label');
            scores_i=net.logp(data_batch);
            scores.append(scores_i.data);
            gt.append(C);
        
        scores=torch.cat(scores,dim=0);
        gt=torch.cat(gt,dim=0);
        
        T=torch.Tensor(1).fill_(0).cuda();
        T.requires_grad_();
        opt2=optim.Adamax([T],lr=3e-2);
        for iter in range(500):
            opt2.zero_grad();
            loss=F.binary_cross_entropy_with_logits(scores*torch.exp(-T),gt.float());
            loss.backward();
            opt2.step();
        
        
        #Eval
        net.eval();
        scores=[];
        scores_pre=[];
        gt=[]
        for data_batch in data_test.batches(max_batch):
            data_batch.cuda();
            
            C=data_batch['label'];
            data_batch.delete_column('label');
            scores_i=net.logp(data_batch);
            
            scores.append((scores_i*torch.exp(-T)).data.cpu());
            scores_pre.append(scores_i.data.cpu());
            
            gt.append(C.data.cpu());
        
        scores=torch.cat(scores,dim=0);
        scores_pre=torch.cat(scores_pre,dim=0);
        gt=torch.cat(gt,dim=0);
        
        def compute_metrics(scores,gt,keys=None):
            if not keys is None:
                #slicing
                categories=set([k for k in keys if not k is None]);
                results={};
                for c in categories:
                    ind=[i for i,k in enumerate(keys) if k==c or k is None];
                    scores_c=[scores[i] for i in ind];
                    gt_c=[gt[i] for i in ind];
                    auc_c,ce_c=compute_metrics(scores_c,gt_c);
                    results[c]={'auc':auc_c,'ce':ce_c};
                
                return results;
            else:
                #Overall
                auc=float(sklearn.metrics.roc_auc_score(torch.LongTensor(gt).numpy(),torch.Tensor(scores).numpy()));
                ce=float(F.binary_cross_entropy_with_logits(torch.Tensor(scores),torch.Tensor(gt)));
                return auc,ce;
        
        auc_i,ce_i=compute_metrics(scores.tolist(),gt.tolist());
        _,ce_pre_i=compute_metrics(scores_pre.tolist(),gt.tolist());
        results_i=compute_metrics(scores.tolist(),gt.tolist(),data_test.data['table_ann']['nclasses'])
        for k in results_i:
            if k in results_by_key:
                results_by_key[k]['auc'].append(results_i[k]['auc'])
                results_by_key[k]['ce'].append(results_i[k]['ce']);
            else:
                results_by_key[k]={'auc':[],'ce':[]};
        
        #results_i=compute_metrics(scores.tolist(),gt.tolist(),data_test.data['table_ann']['arch'])
        #for k in results_i:
        #    if k in results_by_key:
        #        results_by_key[k]['auc'].append(results_i[k]['auc'])
        #        results_by_key[k]['ce'].append(results_i[k]['ce']);
        #    else:
        #        results_by_key[k]={'auc':[],'ce':[]};
        
        results_i=compute_metrics(scores.tolist(),gt.tolist(),data_test.data['table_ann']['trigger'])
        for k in results_i:
            if k in results_by_key:
                results_by_key[k]['auc'].append(results_i[k]['auc'])
                results_by_key[k]['ce'].append(results_i[k]['ce']);
            else:
                results_by_key[k]={'auc':[],'ce':[]};
        
        results_i=compute_metrics(scores.tolist(),gt.tolist(),data_test.data['table_ann']['trigger_subtype'])
        for k in results_i:
            if k in results_by_key:
                results_by_key[k]['auc'].append(results_i[k]['auc'])
                results_by_key[k]['ce'].append(results_i[k]['ce']);
            else:
                results_by_key[k]={'auc':[],'ce':[]};
        
        auc.append(auc_i);
        ce.append(ce_i);
        cepre.append(ce_pre_i);
        session.log('Split %d, loss %.4f (%.4f), auc %.4f, time %f'%(split_id,ce_i,ce_pre_i,auc_i,time.time()-t0));
        
        ensemble.append({'net':net.cpu().state_dict(),'params':params_,'T':float(T.data.cpu())})
        
    
    auc=torch.Tensor(auc);
    ce=torch.Tensor(ce);
    cepre=torch.Tensor(cepre);
    
    if float(auc.mean())>best_loss_so_far:
        best_loss_so_far=float(auc.mean());
        torch.save(ensemble,session.file('model.pt'))
        
    session.log('AUC: %f + %f, CE: %f + %f, CEpre: %f + %f (%s (%d,%d,%d), epochs %d, batch %d, lr %f, decay %f)'%(auc.mean(),2*auc.std(),ce.mean(),2*ce.std(),cepre.mean(),2*cepre.std(),arch,nlayers,nlayers2,nh,epochs,batch,lr,decay));
    
    goal=-float(auc.mean()-2*auc.std());
    
    for k in results_by_key:
        auc=torch.Tensor(results_by_key[k]['auc']);
        ce=torch.Tensor(results_by_key[k]['ce']);
        session.log('\t KEY %s, AUC: %f + %f, CE: %f + %f'%(k,auc.mean(),2*auc.std(),ce.mean(),2*ce.std()));
    
    return goal;



#Get results from hyper parameter search
best=fmin(run_crossval,hp_config,algo=tpe.suggest,max_evals=params.budget)
#best=util.macro.obj(best);
params_=configure_pipeline(**best);
hyper_params_str=json.dumps(best);
session.log('Best hyperparam (%s)'%(hyper_params_str));



#Load extracted features
#fvs_0=torch.load('fvs.pt');
#fvs_1=torch.load('fvs_1.pt');
#fvs=db.union(db.Table.from_rows(fvs_0),db.Table.from_rows(fvs_1));
#fvs.add_index('model_id');

#Load labels
#label=[];
#for i in range(200):
#    fname='/work/projects/trojai-example/data/trojai-round0-dataset/id-%08d/ground_truth.csv'%i;
#    f=open(fname,'r');
#    for line in f:
#        line.rstrip('\n').rstrip('\r')
#        label.append(int(line));
#        break;
#    
#    f.close();

#fvs['label']=label;
#data=db.DB({'table_ann':fvs});
#data.save('data.pt');
