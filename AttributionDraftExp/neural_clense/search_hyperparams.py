from joblib import Parallel, delayed
import queue
import json
import os
import copy
import random
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from utils import init_data_path
from sklearn import metrics

import sys
sys.path.append('/data/isur/0.Work/trojAI/trinityTrojAI_git/users/ksikka/cvpr_exp/troj_round2/')
from utils_temporal_nn.utils_nn import get_splits_random


np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def runner(args):
    worker = q.get()
    gpu = args.gpus[worker % len(args.gpus)]
    print(worker, gpu)

    # Put here your job cmd
    # cmd = "python sweep_pgd.py --model_name %s" % x
    # cmd = "python get_cross_editdist.py --model_name %s" % x

    cmd = "python {}.py --model_name {} --dataset {} --verbose {} \
            --steps {} --attack_succ_thresh {} --save_dir {} --upsample {} --init_cost {} --patience {}".format(
        args.experiment,
        args.model_name,
        args.dataset,
        args.verbose,
        args.steps,
        args.attack_succ_thresh,
        args.save_dir,
        args.upsample,
        args.init_cost,
        args.patience,
    )
    if args.use_kubernetes:
        cmd = cmd + " --use_kubernetes"

    # print (cmd)
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))

    # return gpu id to queue
    q.put(gpu)

def run_list(args, idxs, tag):
    exp_dir = os.path.join(args.save_dir, tag)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    workers = []
    for i in idxs:
        _a = copy.deepcopy(args)
        _a.model_name = i
        _a.save_dir = exp_dir
        workers.append(delayed(runner)(_a))
    Parallel(n_jobs=args.workers, backend="threading")(workers)

def get_measure(idxs, exp_dir, labels):
    results = []
    for i in idxs:
        results_f = os.path.join(exp_dir, i, "result.json")
        if os.path.exists(results_f):
            with open(results_f, "r") as rf:
                data = json.load(rf)
                #results.append(len(data["flag_list"]))
                results.append(int(len(data['flag_list'])>0))
        else:
            results.append(-1)

    results = np.array(results)
    labels = np.array(labels)
    labels = labels[results != -1]
    results = results[results != -1]

    acc = (results == labels).sum() / len(labels)
    fpr, tpr, thresholds = metrics.roc_curve(labels, results)
    auc = metrics.auc(fpr, tpr)

    ret_str = "{}:   auc-{} acc-{} fpr-{} tpr-{} invalid-{}".format(
        tag, auc, acc, fpr, tpr, (results == -1).sum()
    )
    return ret_str, auc, acc

def run_config(args, idxs, labels, tag):
    exp_dir = os.path.join(args.save_dir, tag)
    run_list(args, idxs, tag)

    return get_measure(idxs, exp_dir, labels)

def run_hyperparams(args, meta_data):

    df = meta_data
    idxs_poison = df[df.poisoned == True].model_name.to_list()
    idxs_clean = df[df.poisoned == False].model_name.to_list()
    random.shuffle(idxs_poison)
    random.shuffle(idxs_clean)
    idxs = idxs_poison[: (args.n_models // 2)] + idxs_clean[: (args.n_models // 2)]
    labels = args.n_models // 2 * [1] + args.n_models // 2 * [0]

    # steps_lst       = [20]
    # patience_lst    = [5]
    # thresh_lst      = [0.9]
    # upsample_lst    = [1]
    # init_cost_lst   = [1e-3, 3e-3]
    steps_lst = [20, 50]
    patience_lst = [2, 5]
    thresh_lst = [0.9, 0.99]
    upsample_lst = [1, 2]
    init_cost_lst = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4]

    out_file = os.path.join(args.save_dir, "all_exp.json")
    runs = []
    for steps in steps_lst:
        for patience in patience_lst:
            for attack_succ_thresh in thresh_lst:
                for upsample in upsample_lst:
                    for init_cost in init_cost_lst:
                        try:
                            _a = copy.deepcopy(args)
                            _a.steps = steps
                            _a.patience = patience
                            _a.attack_succ_thresh = attack_succ_thresh
                            _a.upsample = upsample
                            _a.init_cost = init_cost
                            tag = "exp_S-{}_T-{}_U-{}_C-{}_P-{}".format(
                                steps, attack_succ_thresh, upsample, init_cost, patience
                            )
                            runs.append(run_config(_a, idxs, labels, tag))
                        except:
                            runs.append(("{}:   ERROR".format(tag), -1))
                        runs = sorted(runs, key=lambda x: x[1])
                        with open(out_file, "w") as wf:
                            json.dump([en[0] for en in runs], wf, indent=2)

def run_crossvalidation(args, meta_data, tag):
    all_archs = meta_data.model_architecture.unique()
    partitions = []
    for _ in range(args.nfolds):
        partitions.append(
            get_splits_random(meta_data, all_archs, args.test_split, args.val_split)
        )

    test_lst = []
    for p in partitions:
        test_lst += p[-1]
    test_lst = list(set(test_lst))

    config = {en.split('-')[0]:en.split('-')[1] for en in tag.split('_')[1:]}
    args.steps = config['S']
    args.attack_succ_thresh = config['T']
    args.upsample = config['U']
    args.init_cost = config['C']
    args.patience = config['P']
    run_list(args, test_lst, tag)

    test_auc = []
    test_acc = []
    for _p in partitions:
        exp_dir = os.path.join(args.save_dir, tag)
        idxs = _p[-1]
        labels = [int(meta_data[meta_data.model_name==i].iloc[0].poisoned) for i in idxs]
        _, auc, acc = get_measure(idxs, exp_dir, labels)
        test_auc.append(auc)
        test_acc.append(acc)

    return test_auc, test_acc


if __name__ == "__main__":
    per_gpu = 10
    n_gpus = 4
    GPUS = list(range(n_gpus))
    # GPUS = [0,1,2,3,4,5,6,7]
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default="gen_masks_patterns")
    parser.add_argument("--workers", type=int, default=per_gpu * n_gpus)
    parser.add_argument("--gpus", type=int, nargs="+", default=GPUS)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["round1_holdout", "round1", "round2", "mnist"],
    )
    parser.add_argument("--use_kubernetes", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="test/mnist")
    parser.add_argument("--n_models", type=int, default=100)

    parser.add_argument("--test_split", type=int, default=10)
    parser.add_argument("--val_split", type=int, default=10)
    parser.add_argument("--nfolds", type=int, default=5)


    args = parser.parse_args()

    # Put indices in queue
    q = queue.Queue(maxsize=args.workers)
    for i in range(args.workers):
        q.put(i)

    _, meta_data, _ = init_data_path(args.dataset, use_kubernetes=args.use_kubernetes)
    # run_hyperparams(args)

    _results = []
    tags = [
        'exp_S-20_T-0.9_U-2_C-0.01_P-2',   # auc-0.7
        'exp_S-20_T-0.99_U-1_C-0.03_P-5',   # auc-0.7
        'exp_S-20_T-0.9_U-2_C-0.03_P-2',   # auc-0.7
        'exp_S-50_T-0.9_U-2_C-0.01_P-5',   # auc-0.69
        'exp_S-50_T-0.99_U-1_C-0.03_P-2',   # auc-0.69
        'exp_S-20_T-0.99_U-2_C-0.03_P-2',   # auc-0.69
    ]
    for tag in tags:
        res = run_crossvalidation(args, meta_data, tag)
        _results.append((tag,res))

    for tag, (test_auc, test_acc) in _results:
        test_auc_m = np.array(test_auc).mean()
        test_acc_m = np.array(test_acc).mean()
        test_auc_s = np.array(test_auc).std()
        test_acc_s = np.array(test_acc).std()
        print ('{} :'.format(tag))
        print ('ACC: {} +- {}'.format(test_acc_m, test_acc_s))
        print ('AUC: {} +- {}'.format(test_auc_m, test_auc_s))
