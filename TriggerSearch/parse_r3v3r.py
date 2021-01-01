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

def to_cols(fvs):
    ks=list(fvs[0].keys());
    result={};
    for k in ks:
        result[k]=[row[k] for row in fvs];
    
    return result;


#Load extracted features
fvs_color=None;
fvs_polygon=None;
for x in [(0,200),(200,400),(400,600),(600,-1)]:
    fvs_color_x=torch.load('fvs_color_r3_v2_%d_%d.pt'%(x[0],x[1]));
    ids=[int(n[3:]) for n in fvs_color_x['ids']];
    fvs_color_x=db.Table(to_cols(fvs_color_x['fvs']));
    fvs_color_x['model_id']=ids
    if fvs_color is None:
        fvs_color=fvs_color_x;
    else:
        fvs_color=db.union(fvs_color,fvs_color_x);
    
    
    fvs_polygon_x=torch.load('fvs_polygon_r3_v2_%d_%d.pt'%(x[0],x[1]));
    ids=[int(n[3:]) for n in fvs_polygon_x['ids']];
    fvs_polygon_x=db.Table(to_cols(fvs_polygon_x['fvs']));
    fvs_polygon_x['model_id']=ids
    if fvs_polygon is None:
        fvs_polygon=fvs_polygon_x;
    else:
        fvs_polygon=db.union(fvs_polygon,fvs_polygon_x);
    
    print(len(fvs_polygon),len(fvs_color))

fvs_0=db.inner_join(fvs_polygon,fvs_color,'model_id');


fvs_color=torch.load('fvs_color_r3_v2xy.pt');
ids=[int(n[3:]) for n in fvs_color['ids']];
fvs_color=db.Table(to_cols(fvs_color['fvs']));
fvs_color['model_id']=ids
fvs_color.rename_column('color_loss','color_loss_xy');
fvs_color.rename_column('color_loss2','color_loss2_xy');
fvs_color.rename_column('color_brick','color_brick_xy');
fvs_color.rename_column('color_div','color_div_xy');

fvs_0=db.inner_join(fvs_0,fvs_color,'model_id');

fvs_polygon=torch.load('fvs_polygon_r3_v3.pt');
ids=[int(n[3:]) for n in fvs_polygon['ids']];
fvs_polygon=db.Table(to_cols(fvs_polygon['fvs']));
fvs_polygon['model_id']=ids
print(len(fvs_polygon),len(fvs_color))

fvs_0=db.inner_join(fvs_0,fvs_polygon,'model_id');

#Load model meta data
import csv
cols=None;
meta=[];
with open('data/round3-dataset-train/METADATA.csv') as f:
    reader = csv.reader(f);
    for row in reader:
        if cols is None:
            cols=row;
        else:
            meta.append(dict(zip(cols,row)));

meta=db.Table.from_rows(meta);


#Parse meta data into categories
#Num.classes
#Model arch
#Trigger type
model_ids=[];
num_classes_bin=[];
model_arch=[];
trigger_type=[];
trigger_subtype=[];
bgim=[];
adv=[]
ntrig=[];
for i in range(len(meta)):
    id=int(meta[i]['model_name'][3:]); #Hack the int id out
    nclasses=int(meta[i]['number_classes']);
    arch=meta[i]['model_architecture'];
    trigger=meta[i]['trigger_type'];
    ins_subtype=meta[i]['instagram_filter_type'];
    model_ids.append(id);
    num_classes_bin.append('nclasses_%d_%d '%(math.floor((nclasses-5)/7)*7+5,math.floor((nclasses-5)/7)*7+11))
    model_arch.append(arch)
    if trigger=='None':
        trigger_type.append(None);
    else:
        trigger_type.append(trigger);
    
    if trigger=='polygon':
        trigger_subtype.append('polygon');
    elif trigger=='instagram':
        trigger_subtype.append(ins_subtype);
    else:
        trigger_subtype.append(None);
    
    bgim.append(meta[i]['background_image_dataset']);
    adv.append(meta[i]['adversarial_training_method']);
    
    if int(meta[i]['number_triggered_classes_level'])>0:
        ntrig.append('ntrig_%d'%int(meta[i]['number_triggered_classes_level']));
    else:
        ntrig.append(None);
    

meta_=db.Table({'model_id':model_ids,'nclasses':num_classes_bin,'arch':model_arch,'trigger':trigger_type,'trigger_subtype':trigger_subtype,'bgim':bgim,'adv':adv,'ntrig':ntrig});
fvs_0=db.left_join(fvs_0,meta_,'model_id');

print(len(fvs_0['label']))



data=db.DB({'table_ann':fvs_0});
data.save('data_r3v3.pt');
