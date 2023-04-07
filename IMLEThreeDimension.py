import argparse
import os
import io
import itertools
import math
import numpy as np
from PIL import Image
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

import wandb

import plotly.express as px

import Data
import Models
import IO
import LinearProbe
import Utils

device = Utils.device

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

def imle_model_folder(args, make_folder=False):
    data_str = Data.dataset_pretty_name(args.data_tr)
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    lrs = "_".join([f"{lr:.2e}" for idx,lr in enumerate(args.lrs) if idx % 2 == 1])
    folder = f"{args.save_folder}/models_{args.script}/{args.script}-{args.arch}-bs{args.bs}-epochs{args.epochs}-ipe{args.ipe}-lr{lrs}-ns{args.ns}-seed{args.seed}-{args.uid}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, data_tr, data_val, scheduler, args, cur_step, nxz_data_tr=None):
    """Prints evaluation statistics and logs them to WandB.
    
    Args:
    model       -- the model being evaluated
    data_tr     -- ImageFolder-like dataset over training data
    data_val    -- ImageFolder-like dataset over validation data
    scheduler   -- learning rate scheduler for the run
    args        -- argparse Namespace parameterizing run
    cur_step    -- number of training steps so far run
    nxz_data_tr -- ImageLatentDataset over the training data, or None to create
                    it on the fly from [data_tr]
    """
    # Get ImageLatentDatasets as needed
    if nxz_data_tr is None:
        nxz_data_tr = ImageLatentDataset.get_image_latent_dataset(
            model=model,
            loss_fn=nn.MSELoss(reduction="none"),
            dataset=data_tr,
            args=args)
    
    # Generate images
    output_tr = ImageLatentDataset.generate_images(nxz_data_tr, model, args, noise_seed="dataset")

    # Evaluate on the proxy task            
    loss_tr_min = ImageLatentDataset.eval_model(nxz_data_tr, model, args, use_sampled_codes=True)
    loss_tr_mean = ImageLatentDataset.eval_model(nxz_data_tr, model, args, use_sampled_codes=False)

    epoch = int(cur_step / (math.ceil(len(data_tr) / args.bs) * args.ipe) - 1)

    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True)
    
    tqdm.write(f"Epoch {epoch}/{args.epochs} - Step {cur_step}/{len(data_tr) * args.ipe * max(1, args.epochs // args.bs)} - lr={scheduler.get_lr():.5e} loss/min/tr={loss_tr_min:.5e} loss/mean/tr={loss_tr_mean:.5f}")

    # Evaluate on the probing task
    if epoch % args.probe_iter == 0 or epoch == -1 or epoch == args.epochs - 1:
        probe_results = LinearProbe.probe(model, loader_tr, loader_val, args)
    else:
        tqdm.write(f"Computed epoch as {epoch} so not probing")
        probe_results = {}

    wandb.log({
        "loss/min/tr": loss_tr_min,
        "loss/mean/tr": loss_tr_mean,
        "lr": scheduler.get_lr(),
        "train_step": cur_step,
        "embeddings/tr": wandb.Image(output_tr),
        "epoch": epoch,
    }, step=cur_step)

class ImageLatentDataset(Dataset):
    def __init__(self, data, noised_images, latents, images):
        super(ImageLatentDataset, self).__init__()
        self.data = data
        self.noised_images = noised_images.cpu()
        self.latents = latents.cpu()
        self.images = images.cpu()
    
    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.noised_images[idx], self.latents[idx], self.images[idx]
        
    @staticmethod
    def generate_images(nxz_data, model, args, idxs=None, num_images=None,
        num_samples=None, noise_seed=None, latents_seed=None, idxs_seed=None):
        """Returns a PIL image from running [model] on [nxz_data]. Each row in
        the image contains:
        (1) a target image
        (2) a noised image for the target image
        (3) [model] applied to the noised image and a latent code sampled
            for it and the target image
        (4) images given by [model] run on the noised image with random latents
        
        Args:
        nxz_data        -- ImageLatentDataset containing images
        model           -- model to generate images with
        args            -- argparse Namespace with relevant parameters. Its
                            NUM_EVAL_IMAGES, NUM_EVAL_SAMPLES, SEED are overriden by
                            other arguments to this function when they are not None
        idxs            -- indices to [image_latent_data] to use
        num_images      -- number of images to use. Overriden by [idxs], overrides args.num_eval_images if specified
        num_samples     -- number of samples to generate per image, overrides args.num_eval_samples if specified
        noise_seed      -- seed used in sampling image noise if specified, or
                            "dataset" to use the noise in [nxz_data]. Regardless
                            of its value, the noise used in [nxz_data] is used
                            when logging the result of the best latent
        latents_seed    -- seed used in sampling latent codes if specified
        idxs_seed       -- seed used in sampling indices if specified
        """
        data = Data.ZipDataset(nxz_data.data, nxz_data)
        loader = DataLoader(data,
            batch_size=args.code_bs,
            num_workers=args.num_workers,
            pin_memory=True)

        encoder = Utils.de_dataparallel(model).encoder
        encoder = nn.DataParallel(encoder, device_ids=args.gpus).to(device)
        
        with torch.no_grad():
            embeddings, targets = [], []
            for (x,y),(xn,z,_) in tqdm(loader,
                desc=f"Generating embeddings",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                model_input = (x.to(device, non_blocking=True),)
                fxn = encoder(*model_input).cpu()
                embeddings.append(fxn)
                targets.append(y)
        
        embeddings = torch.cat(embeddings, dim=0).cpu().detach().numpy()
        targets = torch.cat(targets, dim=0).view(-1).cpu().detach().numpy()

        fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], color=targets)
        fig.write_image("cur_embeddings.png")
        return Image.open(io.BytesIO(fig.to_image(format="png")))

    @staticmethod
    def eval_model(nxz_data, model, args, loss_fn=None, use_sampled_codes=True):
        """Returns the loss of [model] evaluated on data in nxz_data.

        Args:
        nxz_data            -- ImageLatentDataset
        model               -- IMLE model to evaluate
        args                -- argparse Namespace parameterizing run
        loss_fn             -- loss function to use
        use_sampled_codes   -- whether to use the codes sampled in [nxz_data]
                                (IMLE objective) or random new ones (minimize
                                expected loss)
        """
        loss, total = 0, 0
        loss_fn = nn.MSELoss(reduction="mean") if loss_fn is None else loss_fn
        with torch.no_grad():
            loader = DataLoader(nxz_data,
                batch_size=args.code_bs,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True)
            
            for xn,z,x in tqdm(loader,
                desc="Evaluating on ImageLatentDataset",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                xn = xn.to(device, non_blocking=True)
                z = z.to(device, non_blocking=True) if use_sampled_codes else None
                x = x.to(device, non_blocking=True)
                loss += loss_fn(model(xn, z), x) * len(x)
                total += len(x)
        
        return loss.item() / total

    @staticmethod
    def get_image_latent_dataset(model, loss_fn, dataset, args):
        """Returns an ImageLatentDataset giving noised images and codes for
        [model] to use in IMLE training. 

        Args:
        model   -- IMLE model
        loss_fn -- distance function that returns a BSx... tensor of distances
                    given BSx... inputs. Typically, this means 'reduction' must
                    be 'none'
        dataset -- ImageFolder-like dataset of non-noised images to get codes for
        args    -- argparse Namespace
        """
        if args.zero_half_target == 2:
            return ImageLatentDataset.get_image_latent_dataset_top_and_bottom(model, loss_fn, dataset, args)

        with torch.no_grad():
            least_losses = torch.ones(len(dataset), device=device) * float("inf")
            best_latents = Utils.de_dataparallel(model).get_codes(len(dataset), device=device)
            images = torch.zeros(len(dataset), *dataset[0][0].shape)
            noised_images = torch.zeros(len(dataset), *dataset[0][0].shape)

            loader = DataLoader(dataset,
                batch_size=args.code_bs,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True)

            for idx,(x,y) in tqdm(enumerate(loader),
                desc="Sampling outer loop",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                start_idx = idx * args.code_bs
                stop_idx = min(start_idx + args.code_bs, len(dataset))
                x = x.to(device, non_blocking=True)
                xn = Utils.with_noise(x, std=args.std)

                noised_images[start_idx:stop_idx] = xn.cpu()
                images[start_idx:stop_idx] = x.cpu()

                for sample_idx in tqdm(range(args.ns),
                    desc="Sampling inner loop",
                    leave=False,
                    dynamic_ncols=True):

                    z = Utils.de_dataparallel(model).get_codes(len(xn), device=xn.device)
                    fxn = model(xn, z)
                    losses = loss_fn(fxn, x)
                    losses = torch.sum(losses.view(len(x), -1), dim=1)

                    change_idxs = (losses < least_losses[start_idx:stop_idx])
                    least_losses[start_idx:stop_idx][change_idxs] = losses[change_idxs]
                    best_latents[start_idx:stop_idx][change_idxs] = z[change_idxs]

        return ImageLatentDataset(dataset, noised_images, best_latents.cpu(), images)

def get_args(args=None):
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_data_args(P)
    P = IO.parser_with_logging_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_probe_args(P)
    P = IO.parser_with_imle_args(P)

    args = P.parse_args() if args is None else P.parse_args(args)
    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    args.script = "1D_imle" if args.script is None else args.script
    args.data_tr, args.data_val = "1D", "1D"
    args.lrs = Utils.StepScheduler.process_lrs(args.lrs)
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    if not args.probe_trials == 1:
        raise NotImplementedError(f"Running multiple probe trials is currently not supported in a script that logs to WandB.")
    return args

if __name__ == "__main__":
    args = get_args()    

    if args.resume is None:
        Utils.set_seed(args.seed)
        model = Models.get_model(args, imle=True, in_out_dim=3)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=args.wd)
        last_epoch = -1
    else:
        states = torch.load(args.resume)
        Utils.set_seed(states["seeds"])
        args = argparse.Namespace(**vars(states["args"]) | vars(args))
        model = Models.get_model(args, imle=True,  in_out_dim=3)
        model.load_state_dict(states["model"], strict=False)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=args.wd)
        optimizer.load_state_dict(states["optimizer"])
        model = model.to(device)
        last_epoch = states["epoch"]
        args.uid = states["args"].uid if args.continue_run else args.uid

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL-1D", entity="apex-lab",
        name=os.path.basename(imle_model_folder(args)),
        resume="allow" if args.continue_run else "never",
        settings=wandb.Settings(code_dir=os.path.dirname(__file__)))
    
    scheduler = Utils.StepScheduler(optimizer, args.lrs, last_epoch=last_epoch)
    loss_fn = nn.MSELoss(reduction="mean")
    data_tr, data_val = Data.ThreeDDataset(args), Data.ThreeDDataset(args)
    args.bs = min(args.bs, len(data_tr))

    # data_tr = Subset(data_tr, indices=range(6))
    # data_val = Subset(data_val, indices=range(6))

    tqdm.write(f"---ARGS---\n{Utils.sorted_namespace(args)}\n----------")
    tqdm.write(f"---MODEL---\n{model.module}")
    tqdm.write(f"---OPTIMIZER---\n{optimizer}")
    tqdm.write(f"---SCHEDULER---\n{scheduler}")
    tqdm.write(f"---TRAINING DATA---\n{data_tr}")

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL-1D", entity="apex-lab",
        name=os.path.basename(imle_model_folder(args)),
        resume="allow" if args.continue_run else "never",
        settings=wandb.Settings(code_dir=os.path.dirname(__file__)))

    if not args.save_iter == 0:
        _ = Utils.save_code_under_folder(imle_model_folder(args))

    cur_step = (last_epoch + 1) * args.ipe * math.ceil(len(data_tr) / args.bs)
    if not args.eval_iter == 0:
        _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step)
    for epoch in tqdm(range(last_epoch + 1, args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        epoch_dataset = ImageLatentDataset.get_image_latent_dataset(
            model=model,
            loss_fn=nn.MSELoss(reduction="none"),
            dataset=data_tr,
            args=args)
        epoch_dataset = Data.KKMExpandedDataset(epoch_dataset,
            expand_factor=args.ipe,
            seed=args.seed + epoch)
        loader = DataLoader(epoch_dataset,
            shuffle=False,
            pin_memory=True,
            batch_size=args.bs,
            num_workers=args.num_workers)

        for idx,(xn,z,x) in tqdm(enumerate(loader),
            desc="Batches",
            total=len(loader),
            leave=False,
            dynamic_ncols=True):

            xn = xn.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

            fxn = model(xn, z) 
            loss = loss_fn(fxn, x)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            cur_step += 1

        # Otherwise the worker threads hang around and cause problems?
        del loader
        
        if not args.eval_iter == 0 and (epoch % args.eval_iter == 0
            or epoch == args.epochs - 1):
            _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step,
                nxz_data_tr=epoch_dataset.source)

        if args.save_iter == 0:
            pass
        elif (args.save_iter > 0 and (epoch % args.save_iter == 0
                or epoch in args.save_epochs or epoch == args.epochs -1)):
            _ = Utils.save_state(model, optimizer, args=args, epoch=epoch,
                folder=imle_model_folder(args))
        elif args.save_iter == -1 or args.save_iter > 0:
            _ = Utils.save_state(model, optimizer, args=args, epoch=epoch,
                folder=imle_model_folder(args),
                save_latest=True)

        scheduler.step(epoch)
        
