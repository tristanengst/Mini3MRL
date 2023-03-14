import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import wandb

import Data
import Models
import IO
import LinearProbe
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

def dae_model_folder(args, make_folder=False):
    data_str = Data.dataset_pretty_name(args.data_tr)
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    job_id = "" if args.job_id is None else f"-{args.job_id}"
    lrs = "_".join([f"{lr:.2e}" for idx,lr in enumerate(args.lrs) if idx % 2 == 1])
    folder = f"{args.save_folder}/models_{args.script}/{args.script}-{data_str}-bs{args.bs}-epochs{args.epochs}-lr{lrs}-nshot{args.n_shot}-nway{args.n_way}-std{args.std}-{args.seed}-{args.uid}{job_id}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, loader_tr, loader_val, scheduler, args, cur_step):
    eval_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    loss_tr, loss_val = 0, 0
    total_tr, total_val = 0, 0
    with torch.no_grad():
        for x_tr,_ in loader_tr:
            x_tr = x_tr.to(device, non_blocking=True)
            nx_tr = Utils.with_noise(x_tr, std=args.std)
            fxn_tr = model(nx_tr)
            loss_tr += eval_loss_fn(fxn_tr, x_tr)
            total_tr += len(x_tr)

        for x_val,_ in loader_val:
            x_val = x_val.to(device, non_blocking=True)
            nx_val = Utils.with_noise(x_val, std=args.std)
            fxn_val = model(nx_val)
            loss_val += eval_loss_fn(fxn_val, x_val)
            total_val += len(x_val)
    
    loss_tr = loss_tr.item() / total_tr
    loss_val = loss_val.item() / total_val

    epoch = cur_step // len(loader_tr)
    if epoch % args.probe_iter == 0 or epoch == args.epochs - 1:
        acc_vals = LinearProbe.probe(model, loader_tr, loader_val, args)
        acc_vals_str = " ".join([f"{k}={v:.5f}" for k,v in acc_vals.items()])
    else:
        acc_vals = {}
        acc_vals_str = ""
    
    tqdm.write(f"Step {cur_step}/{len(loader_tr) * args.epochs} - lr={scheduler.get_lr():.5e} loss/tr={loss_tr:.5f} loss/te={loss_val:.5f} {acc_vals_str}")

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

    wandb.log(acc_vals | {
        "loss/te": loss_val,
        "loss/tr": loss_tr,
        "lr": scheduler.get_lr(),
        "train_step": cur_step,
        "images/te": wandb.Image(image_path_val),
        "images/tr": wandb.Image(image_path_tr),
        "epoch": cur_step // (len(loader_tr))
    }, step=cur_step)

    

def get_args(args=None):
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_data_args(P)
    P = IO.parser_with_logging_args(P)
    P = IO.parser_with_probe_args(P)

    args = P.parse_args() if args is None else P.parse_args(args)
    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    args.script = "dae" if args.script is None else args.script
    args.lrs = Utils.StepScheduler.process_lrs(args.lrs)
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    if not args.num_eval_samples == 1:
        tqdm.write(f"LOG: setting NUM_EVAL_SAMPLES to 1")
        args.num_eval_samples = 1

    assert args.probe_iter % args.eval_iter == 0

    return args

if __name__ == "__main__":
    args = get_args()

    if args.resume is None:
        Utils.set_seed(args.seed)
        model = Models.get_model(args, imle=False)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=1e-5)
        last_epoch = -1
    else:
        states = torch.load(args.resume)
        Utils.set_seed(states["seeds"])
        args = argparse.Namespace(**vars(states["args"]) | vars(args))
        model = Models.get_model(args, imle=False)
        model.load_state_dict(states["model"])
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=1e-5)
        optimizer.load_state_dict(states["optimizer"])
        model = model.to(device)
        last_epoch = states["epoch"]

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL", entity="apex-lab",
        settings=wandb.Settings(code_dir="."),
        name=os.path.basename(dae_model_folder(args)))
    
    scheduler = Utils.StepScheduler(optimizer, args.lrs)
    loss_fn = nn.BCEWithLogitsLoss()

    data_tr, data_val = Data.get_data_from_args(args)
        
    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_val}")
    
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True)
    
    tqdm.write(f"-------\n{Utils.sorted_namespace(args)}\n-------")
    tqdm.write(f"Will save to {dae_model_folder(args)}")

    cur_step = (last_epoch + 1) * len(loader_tr)
    _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
    
    for epoch in tqdm(range(args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        for x,_ in tqdm(loader_tr,
            desc="Batches",
            dynamic_ncols=True,
            leave=False):

            x = x.to(device, non_blocking=True)
            nx = Utils.with_noise(x, std=args.std)
            
            fx = model(nx)
            loss = loss_fn(fx, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            cur_step += 1
        
        _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
        
        if not args.save_iter == 0 and epoch % args.save_iter == 0:
            _ = Utils.save_state(model, optimizer,
                args=args,
                epoch=epoch,
                folder=dae_model_folder(args))
        elif args.save_iter == -1:
            raise NotImplementedError()
    
        scheduler.step(epoch)



        
