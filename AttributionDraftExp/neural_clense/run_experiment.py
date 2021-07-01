from joblib import Parallel, delayed
import queue
import os
from argparse import ArgumentParser
import pandas as pd
from random import shuffle
from utils import init_data_path


def runner(x):
    worker = q.get()
    gpu = args.gpus[worker % len(args.gpus)]
    print (x, worker, gpu)

    # Put here your job cmd
    # cmd = "python sweep_pgd.py --model_name %s" % x
    # cmd = "python get_cross_editdist.py --model_name %s" % x

    cmd = "python {}.py --model_name {} --dataset {} --verbose {} --steps {} --attack_succ_thresh {} ".format(args.experiment,
                                                            x, args.dataset, args.verbose, args.steps, args.attack_succ_thresh)
    if args.use_kubernetes:
        cmd = cmd+' --use_kubernetes'

    #print (cmd)
    os.system("CUDA_VISIBLE_DEVICES=%d %s" % (gpu, cmd))

    # return gpu id to queue
    q.put(gpu)

if __name__ == "__main__":
    per_gpu = 4
    n_gpus  = 4
    GPUS = list(range(n_gpus))
    # GPUS = [0,1,2,3,4,5,6,7]
    parser = ArgumentParser()
    parser.add_argument("--experiment", type=str, default="gen_masks_patterns")
    parser.add_argument("--workers", type=int, default=per_gpu*n_gpus)
    parser.add_argument("--gpus", type=int, nargs='+', default=GPUS)
    parser.add_argument("--dataset", type=str, default="mnist", choices=['round1_holdout', 'round1', 'round2', 'mnist'])
    parser.add_argument("--use_kubernetes", action="store_true")

    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--attack_succ_thresh", type=float, default=0.9)

    args = parser.parse_args()


    # Put indices in queue
    q = queue.Queue(maxsize=args.workers)
    for i in range(args.workers):
        q.put(i)


    #meta_data_f = os.path.join(args.src_dir, 'METADATA.csv')
    #meta_data = pd.read_csv(meta_data_f)

    _, meta_data, _ = init_data_path(args.dataset, use_kubernetes=args.use_kubernetes)

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
