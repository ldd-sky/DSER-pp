import argparse
import random
import numpy as np
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from core.train import train
from model.ourModel import Model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=6, type=int, help='minibatch size')
    parser.add_argument('--test_batch_size', default=8, type=int, help='test minibatch size')
    parser.add_argument('--num_worker', default=4, type=int, help='num worker')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    parser.add_argument('--RESUME', default=False, type=bool, help='RESUME')
    parser.add_argument('--RESUME_EPOCH', default=20, type=int, help='RESUME_EPOCH')
    parser.add_argument('--train_dateset', default='vimeo', type=str, help='vimeo/hqf/hsergb')
    parser.add_argument('--sample_factor', default=2, type=int, help='sampling proportion, 1/factor')
    parser.add_argument('--mask_patch_size', default=32, type=int, help='different mask patch size')
    args = parser.parse_args()
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    model = Model(args)
    train(model, args)
