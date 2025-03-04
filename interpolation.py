import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import warnings

from predict.bsergb import predict_bsergb
from predict.gopro import predict_gopro
from predict.hsergb import predict_hsergb
from predict.sunfilm import predict_snufilm
from predict.vimeo90k import predict_vimeo90k
from model.ourModel import Model

device = torch.device("cuda")
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='bsergb', type=str, help='dataset name')
    parser.add_argument('--bins', default=8, type=int, help='number of time bins')
    parser.add_argument('--sample_factor', default=1, type=int, help='sampling proportion, 1/factor')
    parser.add_argument('--mask_patch_size', default=32, type=int, help='different mask patch size')
    args = parser.parse_args()
    model = Model(args)

    model.load_checkpoint("xxxx")

    val_root_path = "xxxx"
    if args.dataset == 'vimeo90k':
        save_path = val_root_path + "90k"
        saveIndexRange = [0, 2000]
        predict_vimeo90k(model, args.bins, device, save_path, isSave=False, isTestPer=True,
                         saveIndexRange=saveIndexRange)
    if args.dataset == 'gopro':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_gopro(model=model, bins=args.bins, device=device, save_path=save_path, multis=[7, 15]
                      , isSave=False, isTestPer=False, saveSpecificScene=saveSpecificScene)
    if args.dataset == 'snufilm':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_snufilm(model, args.bins, device, save_path, difficulties=['extreme', 'hard'], isSave=False,
                        isTestPer=False,
                        saveSpecificScene=saveSpecificScene)
    if args.dataset == 'hsergb':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_hsergb(model, args.bins, device, save_path, multis=[5, 7], isSave=False, isTestPer=False,
                       saveSpecificScene=saveSpecificScene)

    if args.dataset == 'bsergb':
        save_path = val_root_path + args.dataset
        saveSpecificScene = None
        predict_bsergb(model, args.bins, device, save_path, multis=[1, 3], isSave=True, isTestPer=False,
                       saveSpecificScene=saveSpecificScene)
