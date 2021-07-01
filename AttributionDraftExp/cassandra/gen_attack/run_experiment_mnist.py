from joblib import Parallel, delayed
import queue
import os
from argparse import ArgumentParser
import pandas as pd
from random import shuffle


def runner(x):
    worker = q.get()
    gpu = args.gpus[worker % len(args.gpus)]
    print (x, worker, gpu)

    # Put here your job cmd
    # cmd = "python sweep_pgd.py --model_name %s" % x
    # cmd = "python get_cross_editdist.py --model_name %s" % x

    cmd = "python {}.py --model_name {} --round {} --src_dir {} --checkpoint_dir {} --env_name {} --attack_type {} ".format(args.experiment,
                                                            x, args.round, args.src_dir, args.checkpoint_dir, args.env_name, args.attack_type)

    #print (cmd)
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))

    # return gpu id to queue
    q.put(gpu)

if __name__ == "__main__":
    per_gpu = 6
    n_gpus  = 4
    GPUS = list(range(n_gpus))
    # GPUS = [0,1,2,3,4,5,6,7]
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default="launch_attack_mnist")
    parser.add_argument("--workers", type=int, default=per_gpu*n_gpus)
    parser.add_argument("--gpus", type=int, nargs='+', default=GPUS)

    # parser.add_argument("--round", type=int, default=2)
    # parser.add_argument("--src_dir", type=str, default="/data/datasets/round2-dataset-train/")
    # parser.add_argument("--checkpoint_dir", type=str, default="/code/unv_pert_feat/round2")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--src_dir", type=str, default="/data/ksikka/projects_2020/trojAI/mnist-dataset/")
    parser.add_argument("--checkpoint_dir", type=str, default="/data/isur/0.Work/trojAI/trinityTrojAI/users/ksikka/checkpoints_path_bkp/unv_pert_feat/mnist")
    # parser.add_argument("--src_dir", type=str, default="/data/datasets/round1-dataset-train/")
    # parser.add_argument("--checkpoint_dir", type=str, default="/code/unv_pert_feat/round1")

    # parser.add_argument("--env_name", type=str, default="univ_pert_ut_l2")
    # parser.add_argument("--attack_type", type=str, default="univ_pert_ut_2")
    parser.add_argument("--env_name", type=str, default="univ_pert_ut_l1")
    parser.add_argument("--attack_type", type=str, default="univ_pert_ut_1")

    args = parser.parse_args()


    # Put indices in queue
    q = queue.Queue(maxsize=args.workers)
    for i in range(args.workers):
        q.put(i)


    meta_data_f = os.path.join(args.src_dir, 'METADATA.csv')
    meta_data = pd.read_csv(meta_data_f)

    #filter_arch = ['googlenet', 'mobilenetv2', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'vgg11bn', 'vgg13bn', 'vgg16bn', 'vgg19bn', 'wideresnet101', 'wideresnet50']
    #filter_arch = ['inceptionv3']
    #df = meta_data[meta_data.model_architecture.isin(filter_arch)]
    df = meta_data
    #idxs = df.index.to_list()
    idxs = df.model_name.to_list()
    #import ipdb; ipdb.set_trace()
    #models= meta_data.model_name.to_list()
    # models.reverse()
    shuffle(idxs)
    workers = [delayed(runner)(i) for i in idxs]
    Parallel(n_jobs=args.workers, backend="threading")(workers)
