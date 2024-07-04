import torch
import numpy as np
import argparse
from trainer import training
from data_loader import dataLoader
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


if __name__ == '__main__':

    same_seeds(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--dataset', type=str, default='Dunnhumby_sample')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--l2', type=float, default=1e-3)
    parser.add_argument('--Lh', type=int, default=5)
    parser.add_argument('--Lv', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--dim', type=int, default=32)
    parser.add_argument('--isTrain', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--evalEpoch', type=int, default=1)
    parser.add_argument('--testOrder', type=int, default=1)
    parser.add_argument('--ac_conv', type=str, default='relu')
    parser.add_argument('--ac_fc', type=str, default='relu')
    parser.add_argument('--drop', type=float, default=0.5)
    parser.add_argument('--maxLenSeq', type=int, default=30)
    parser.add_argument('--maxLenBas', type=int, default=30)
    config = parser.parse_args()

    print(config)

    dataset = dataLoader(config)
    if config.isTrain:
        config.padIdx = dataset.numItemsTrain
    else:
        config.padIdx = dataset.numItemsTest

    print('start training')
    training(dataset, config, device)
