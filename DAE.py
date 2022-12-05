import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import wandb

import Models
import IO
import LinearProbe
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dae_model_folder(args, make_folder=True):
    subsample = "_all" if args.subsample is None else args.subsample
    suffix = "" if args.suffix is None else f"-{arg.suffix}"
    folder = f"{args.save_folder}/dae_models/imle-bs{args.bs}-epochs{args.epochs}-lr{args.bs}-std{args.std}-subsample{subsample}seed{args.seed}-{args.uid}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, loader_tr, loader_te, scheduler, args, epoch=0):
    eval_loss_fn = nn.MSELoss(reduction="sum")
    loss_tr, loss_te = 0, 0
    total_tr, total_te = 0, 0
    for x,_ in loader_tr:
        x = x.to(device, non_blocking=True)
        x = x.view(x.shape[0], -1)
        nx_tr = Utils.with_noise(x, std=args.std)
        fxn_tr = model(nx_tr)
        loss_tr += eval_loss_fn(fxn_tr, x)
        total_tr += len(x)

    for x_te,_ in loader_te:
        x_te = x_te.to(device, non_blocking=True)
        x_te = x_te.view(x.shape[0], -1)
        nx_te = Utils.with_noise(x_te, std=args.std)
        fxn_te = model(nx_te)
        loss_te += eval_loss_fn(fxn_te, x_te)
        total_te += len(x_te)
    
    loss_tr = loss_tr.item() / total_tr
    loss_te = loss_te.item() / total_te

    acc_te = LinearProbe.linear_probe(model, loader_tr, loader_te, args)
    tqdm.write(f"Epoch {epoch:5}/{args.epochs} step {epoch * len(loader_tr):6}/{len(loader_tr) * args.epochs} - lr={scheduler.get_last_lr()[0]:.5e} loss/tr={loss_tr:.5f} loss/te={loss_te:.5f} acc/te={acc_te:.5f}")

    # Create an image to visualize how the model is doing
    image_save_folder = f"{dae_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    image_path = f"{image_save_folder}/{epoch}_te.png"
    image = torch.cat([x_te[:8].unsqueeze(1), nx_te[:8].unsqueeze(1), fxn_te[:8].unsqueeze(1)], dim=1)
    Utils.images_to_pil_image(image).save(image_path)

    wandb.log({
        "acc/te": acc_te,
        "loss/te": loss_te,
        "loss/tr": loss_tr,
        "lr": scheduler.get_last_lr()[0],
        "train_step": epoch * len(loader_tr),
        "images/te": wandb.Image(image_path)
    })

def get_args():
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_probe_args(P)

    args = P.parse_args()
    args.uid = wandb.util.generate_id()
    return args

if __name__ == "__main__":
    args = get_args()
    tqdm.write(str(args))
    Utils.set_seed(args.seed)

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL",
        name=os.path.basename(dae_model_folder(args)))

    tqdm.write(f"Will save to {dae_model_folder(args)}")


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
        persistent_workers=False,
        pin_memory=True)
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=20,
        persistent_workers=False,
        pin_memory=True)
    
    model = Models.DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=args.epochs // 5,
        gamma=.3)

    _ = evaluate(model, loader_tr, loader_te, scheduler, args, epoch=0)

    for epoch in tqdm(range(args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        for x,_ in loader_tr:

            x = x.to(device, non_blocking=True)
            x = x.view(x.shape[0], -1)
            nx = Utils.with_noise(x, std=args.std)

            fx = model(nx)
            loss = loss_fn(fx, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if epoch % args.eval_iter == 0 or epoch == args.epochs - 1:
            _ = evaluate(model, loader_tr, loader_te, scheduler, args, epoch+1)
        scheduler.step() 
        



        
