
import torch
import os
import numpy
import time
import sklearn.metrics
import importlib
import torch.nn.functional as F
import math


def demo_trojan_detector(model_filepath, result_filepath=None, scratch_dirpath=None, examples_dirpath=None, example_img_format='png'):
    try:
        checkpoint=torch.load('session_0000410/model.pt');
    except:
        checkpoint=torch.load('/session_0000410/model.pt');
    
    import extract_fv_color_v2r as extract_fv_color
    fvs_color=extract_fv_color.extract_fv(model_filepath,examples_dirpath)
    import extract_fv_color_v2xy as extract_fv_color
    fvs_color_xy=extract_fv_color.extract_fv(model_filepath,examples_dirpath)
    import extract_fv_v17r as extract_fv
    fvs=extract_fv.extract_fv(model_filepath,examples_dirpath)
    
    
    for k in fvs_color:
        fvs[k]=fvs_color[k];
    
    fvs['color_loss_xy']=fvs_color_xy['color_loss']
    fvs['color_loss2_xy']=fvs_color_xy['color_loss2']
    fvs['color_brick_xy']=fvs_color_xy['color_brick']
    fvs['color_div_xy']=fvs_color_xy['color_div']
    
    s=[];
    
    for i in range(len(checkpoint)):
        params_=checkpoint[i]['params'];
        arch_=importlib.import_module(params_.arch);
        net=arch_.new(params_);
        
        net.load_state_dict(checkpoint[i]['net']);
        net=net.cuda();
        net.eval();
        
        s_i=(net.logp(fvs)*math.exp(-checkpoint[i]['T'])).data.cpu();
        s.append(float(s_i))
    
    s=sum(s)/len(s);
    s=torch.sigmoid(torch.Tensor([s]));
    
    trojan_probability = float(s);
    print('Trojan Probability: {}'.format(trojan_probability))
    
    #with open(result_filepath, 'w') as fh:
    #    fh.write("{}".format(trojan_probability))
    
    return trojan_probability;
    


#Task: try to identify trojan triggers from images
if __name__ == "__main__":
    root='data/round2-dataset-train/models';
    models=os.listdir(root);
    models=sorted(models);


    t0=time.time();
    pred=[];
    gt=[];
    for id in range(1100,1000,-1):
        fname=os.path.join(root,models[id],'ground_truth.csv');
        f=open(fname,'r');
        for line in f:
            line.rstrip('\n').rstrip('\r')
            label=int(line);
            break;
        
        f.close();
        
        model_filepath=os.path.join(root,models[id],'model.pt');
        examples_dirpath=os.path.join(root,models[id],'example_data');
        s=demo_trojan_detector(model_filepath=model_filepath,examples_dirpath=examples_dirpath);
        pred.append(s);
        gt.append(label);
        
        if id<=1095:
            auc=sklearn.metrics.roc_auc_score(torch.LongTensor(gt).numpy(),torch.Tensor(pred).numpy());
            loss=float(F.binary_cross_entropy(torch.Tensor(pred),torch.Tensor(gt)));
            
            print('model %d, loss %.4f, auc %.4f, time %f'%(id,loss,auc,time.time()-t0));


