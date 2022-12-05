import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import Models
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
    tqdm.write(str(args))
    Utils.set_seed(args.seed)

    data_tr = MNIST("./data/",
        download=True,
        train=True,
        transform=Utils.transforms_tr)

    # Assumes the data is shuffled, which is true for the MNIST
    if not args.subsample is None:
        data_tr = Subset(data_tr, indices=range(args.subsample))
    else:
        data_tr = data_tr
    
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
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=20,
        persistent_workers=True,
        pin_memory=True)
    
    model = Models.DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=args.epochs // 5,
        gamma=.3)

    for epoch in tqdm(range(args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        for x,_ in loader_tr:

            x = x.to(device, non_blocking=True)
            x = x.view(x.shape[0], -1)
            nx = Utils.with_noise(x)

            fx = model(nx)
            loss = loss_fn(fx, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


        scheduler.step()
        if epoch % args.eval_iter == 0 or epoch == args.epochs - 1:

            eval_loss_fn = nn.MSELoss(reduction="sum")
            loss_tr, loss_te = 0, 0
            total_tr, total_te = 0, 0
            for x,_ in loader_tr:
                x = x.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                nx = Utils.with_noise(x)
                fx = model(nx)
                loss_tr += eval_loss_fn(fx, x)
                total_tr += len(x)

            for x,_ in loader_te:
                x = x.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                nx = Utils.with_noise(x)
                fx = model(nx)
                loss_te += eval_loss_fn(fx, x)
                total_te += len(x)
            
            loss_tr = loss_tr.item() / total_tr
            loss_te = loss_te.item() / total_te

            acc_te = LinearProbe.linear_probe(model, loader_tr, loader_te, args)
            tqdm.write(f"Epoch {epoch+1:5}/{args.epochs} - lr={scheduler.get_last_lr()[0]:.5e} loss/tr={loss_tr:.5f} loss/te={loss_te:.5f} acc/te={acc_te:.5f}")    
        



        
