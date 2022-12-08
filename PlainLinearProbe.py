import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import IO
import LinearProbe
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_name(args): pass

def get_args():
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_probe_args(P)

    args = P.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    data_tr = MNIST("./data/",
        download=True,
        train=True,
        transform=Utils.transforms_tr)
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=20,
        persistent_workers=True,
        pin_memory=True)
    data_te = MNIST("./data/",
        download=True,
        train=False,
        transform=Utils.transforms_tr)
    loader_te = DataLoader(data_te,
        batch_size=args.bs,
        shuffle=True,
        num_workers=20,
        persistent_workers=True,
        pin_memory=True)

    acc_te = LinearProbe.plain_linear_probe(loader_tr, loader_te, args)
    tqdm.write(f"Accuracy: {acc_te:.5f}")

        
