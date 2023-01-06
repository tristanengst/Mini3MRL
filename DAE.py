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
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    job_id = "" if args.job_id is None else f"-{args.job_id}
    folder = f"{args.save_folder}/dae_models/dae-bs{args.bs}-epochs{args.epochs}-lr{args.bs}-std{args.std}-subsample{subsample}seed{args.seed}-{args.uid}{job_id}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, loader_tr, loader_val, scheduler, args, cur_step):
    eval_loss_fn = nn.MSELoss(reduction="sum")
    loss_tr, loss_val = 0, 0
    total_tr, total_val = 0, 0
    with torch.no_grad():
        for x_tr,_ in loader_tr:
            x_tr = x_tr.to(device, non_blocking=True).view(x.shape[0], -1)
            nx_tr = Utils.with_noise(x_tr, std=args.std)
            fxn_tr = model(nx_tr)
            loss_tr += eval_loss_fn(fxn_tr, x_tr)
            total_tr += len(x_tr)

        for x_val,_ in loader_val:
            x_val = x_val.to(device, non_blocking=True).view(x.shape[0], -1)
            nx_val = Utils.with_noise(x_val, std=args.std)
            fxn_val = model(nx_val)
            loss_val += eval_loss_fn(fxn_val, x_val)
            total_val += len(x_val)
    
    loss_tr = loss_tr.item() / total_tr
    loss_val = loss_val.item() / total_val

    acc_val = LinearProbe.linear_probe(model, loader_tr, loader_val, args)
    tqdm.write(f"Step {cur_step:6}/{len(loader_tr) * args.epochs} - lr={scheduler.get_last_lr()[0]:.5e} loss/tr={loss_tr:.5f} loss/te={loss_val:.5f} acc/te={acc_val:.5f}")

    # Create an image to visualize how the model is doing
    image_save_folder = f"{dae_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    image_path_tr = f"{image_save_folder}/{cur_step}_tr.png"
    image = torch.cat([x_tr[:8].unsqueeze(1), nx_tr[:8].unsqueeze(1), fxn_tr[:8].unsqueeze(1)], dim=1)
    Utils.images_to_pil_image(image).save(image_path_tr)

    # Create an image to visualize how the model is doing
    image_save_folder = f"{dae_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    image_path_val = f"{image_save_folder}/{cur_step}_val.png"
    image = torch.cat([x_val[:8].unsqueeze(1), nx_val[:8].unsqueeze(1), fxn_val[:8].unsqueeze(1)], dim=1)
    Utils.images_to_pil_image(image).save(image_path_val)

    wandb.log({
        "acc/te": acc_val,
        "loss/te": loss_val,
        "loss/tr": loss_tr,
        "lr": scheduler.get_last_lr()[0],
        "train_step": cur_step,
        "images/tr": wandb.Image(image_path_tr),
        "images/te": wandb.Image(image_path_val)
    })

def get_args():
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_data_args(P)
    P = IO.parser_with_logging_args(P)
    P = IO.parser_with_probe_args(P)

    args = P.parse_args() if args is None else P.parse_args(args)
    args.uid = wandb.util.generate_id() if args.wandb is None else args.wandb
    args.script = "DAE" if args.script is None else args.script
    return args

if __name__ == "__main__":
    args = get_args()
    tqdm.write(str(args))
    Utils.set_seed(args.seed)

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL",
        name=os.path.basename(dae_model_folder(args)))

    tqdm.write(f"Will save to {dae_model_folder(args)}")

    # Get DataLoaders over the training and testing data
    data = get_dataset(args.data_tr, split="train", Utils.transforms_tr)
    data_tr = Data.get_fewshot_dataset(data, n_way=args.n_way, n_shot=args.n_shot, seed=args.seed)
    data_val = Data.ImageFolderSubset.complement(data_tr)
    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_tr}")
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True)
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=True)
    
    model = Models.DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=args.epochs // 5,
        gamma=.3)

    cur_step = 0
    num_steps = args.ipe * len(loader_tr) * args.epochs
    log_iter = max(1, args.epochs // args.evals)
    _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
    for epoch in tqdm(range(args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        for x,_ in loader_tr:

            x = x.to(device, non_blocking=True).view(x.shape[0], -1)
            nx = Utils.with_noise(x, std=args.std)

            fx = model(nx)
            loss = loss_fn(fx, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            cur_step += 1
        
        if epoch % log_iter == 0 or epoch == args.epochs - 1:
            _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
    
        scheduler.step()
        



        
