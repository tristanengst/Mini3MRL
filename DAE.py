import argparse
import os
import itertools
import math
import numpy as np
import sys
import time
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

device = Utils.device

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

def dae_model_folder(args, make_folder=False):
    data_str = Data.dataset_pretty_name(args.data_tr)
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    job_id = "" if args.job_id is None else f"-{args.job_id}"
    lrs = "_".join([f"{lr:.2e}" for idx,lr in enumerate(args.lrs) if idx % 2 == 1])
    folder = f"{args.save_folder}/models_{args.script}/{args.script}-{data_str}-bs{args.bs}-epochs{args.epochs}-feat_dim{args.feat_dim}-lr{lrs}-nshot{args.n_shot}-nway{args.n_way}-seed{args.seed}-{args.uid}{job_id}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, data_tr, data_val, scheduler, args, cur_step, nx_data_tr=None):
    """Prints evaluation statistics and logs them to WandB.
    
    Args:
    model       -- the model being evaluated
    data_tr     -- ImageFolder-like dataset over training data
    data_val    -- ImageFolder-like dataset over validation data
    scheduler   -- learning rate scheduler for the run
    args        -- argparse Namespace parameterizing run
    cur_step    -- number of training steps so far run
    nx_data_tr -- ImageDataset over the training data, or None to create
                    it on the fly from [data_tr]
    """
    # Get ImageDatasets as needed
    if nx_data_tr is None:
        nx_data_tr = ImageDataset.get_image_dataset(
            model=model,
            dataset=data_tr,
            args=args)
    nx_data_val = ImageDataset.get_image_dataset(
        model=model,
        dataset=data_val,
        args=args)

    # # Generate embeddings.
    # embeds_pre_fuse_tr, targets_pre_fuse_tr = ImageDataset.generate_embeddings(nx_data_tr, model, args, mode="pre_fuse")
    # embeds_pre_fuse_val, targets_pre_fuse_val = ImageDataset.generate_embeddings(nx_data_val, model, args, mode="pre_fuse")
    # embeds_no_noise_tr, targets_no_noise_tr = ImageDataset.generate_embeddings(nx_data_tr, model, args, mode="no_noise")
    # embeds_no_noise_val, targets_no_noise_val = ImageDataset.generate_embeddings(nx_data_val, model, args, mode="no_noise")

    # if args.feat_dim < 3:
    #     embedding_visualization = {
    #         "embeds/pre_fuse_vis/tr": wandb.Image(Utils.embeddings_to_pil_image(embeds_pre_fuse_tr, targets_pre_fuse_tr)),
    #         "embeds/pre_fuse_vis/val": wandb.Image(Utils.embeddings_to_pil_image(embeds_pre_fuse_val, targets_pre_fuse_val)),
    #         "embeds/no_noise_vis/tr": wandb.Image(Utils.embeddings_to_pil_image(embeds_no_noise_tr, targets_no_noise_tr)),
    #         "embeds/no_noise_vis/val": wandb.Image(Utils.embeddings_to_pil_image(embeds_no_noise_val, targets_no_noise_val))}
    # else:
    #     embedding_visualization = {}

    # embedding_results = {
    #     "embeds/pre_fuse_mean/tr": torch.mean(embeds_pre_fuse_tr),
    #     "embeds/pre_fuse_mean/val": torch.mean(embeds_pre_fuse_val),
    #     "embeds/no_noise_mean/tr": torch.mean(embeds_no_noise_tr),
    #     "embeds/no_noise_mean/val": torch.mean(embeds_no_noise_val),

    #     "embeds/pre_fuse_feat_std/tr": torch.mean(torch.std(embeds_pre_fuse_tr, dim=0)),
    #     "embeds/pre_fuse_feat_std/val": torch.mean(torch.std(embeds_pre_fuse_val, dim=0)),
    #     "embeds/no_noise_feat_std/tr": torch.mean(torch.std(embeds_no_noise_tr, dim=0)),
    #     "embeds/no_noise_feat_std/val": torch.mean(torch.std(embeds_no_noise_val, dim=0)),

    #     "embeds/pre_fuse_ex_std/tr": torch.mean(torch.std(embeds_pre_fuse_tr, dim=1)),
    #     "embeds/pre_fuse_ex_std/val": torch.mean(torch.std(embeds_pre_fuse_val, dim=1)),
    #     "embeds/no_noise_ex_std/tr": torch.mean(torch.std(embeds_no_noise_tr, dim=1)),
    #     "embeds/no_noise_ex_std/val": torch.mean(torch.std(embeds_no_noise_val, dim=1)),

    #     "embeds/pre_fuse_abs/tr": torch.mean(torch.abs(embeds_pre_fuse_tr)),
    #     "embeds/pre_fuse_abs/val": torch.mean(torch.abs(embeds_pre_fuse_val)),
    #     "embeds/no_noise_abs/tr": torch.mean(torch.abs(embeds_no_noise_tr)),
    #     "embeds/no_noise_abs/val": torch.mean(torch.abs(embeds_no_noise_val)),
    # }
    
    # Generate images
    images_tr = ImageDataset.generate_images(nx_data_tr, model, args)
    images_val = ImageDataset.generate_images(nx_data_val, model, args)
    # if args.save_iter > 0:
    #     image_save_folder = f"{dae_model_folder(args, make_folder=True)}/images"
    #     Utils.conditional_make_folder(image_save_folder)
    #     images_tr.save(f"{image_save_folder}/{cur_step}_tr.png")
    #     images_val.save(f"{image_save_folder}/{cur_step}_val.png")
    image_save_folder = f"{dae_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    images_tr.save(f"{image_save_folder}/{cur_step}_tr.png")
    images_val.save(f"{image_save_folder}/{cur_step}_val.png")

    # Evaluate on the proxy task
    loss_tr_mean = ImageDataset.eval_model(nx_data_tr, model, args)
    loss_val_mean = ImageDataset.eval_model(nx_data_val, model, args)

    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True)

    epoch = (cur_step // (len(loader_tr))) - 1

    tqdm.write(f"Epoch {epoch}/{args.epochs} - Step {cur_step}/{len(loader_tr) * args.epochs} - lr={scheduler.get_lr():.5e} loss/mean/tr={loss_tr_mean:.5f} loss/mean/val={loss_val_mean:.5f}")

    # Evaluate on the probing task
    if epoch % args.probe_iter == 0 or epoch == -1 or epoch == args.epochs - 1:
        probe_results = LinearProbe.probe(model, loader_tr, loader_val, args)
    else:
        tqdm.write(f"Computed epoch as {epoch} so not probing")
        probe_results = {}

    # wandb.log(probe_results | embedding_results | embedding_visualization | {
    #     "loss/mean/tr": loss_tr_mean,
    #     "loss/mean/val": loss_val_mean,
    #     "lr": scheduler.get_lr(),
    #     "train_step": cur_step,
    #     "images/val": wandb.Image(images_val),
    #     "images/tr": wandb.Image(images_tr),
    #     "epoch": epoch,
    # }, step=cur_step)

class ImageDataset(Dataset):
    def __init__(self, data, noised_images, images):
        super(ImageDataset, self).__init__()
        self.data = data
        self.noised_images = noised_images.cpu()
        self.images = images.cpu()
    
    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.noised_images[idx], self.images[idx]

    @staticmethod
    def generate_embeddings(nx_data, model, args, mode="pre_fuse", k=1000):
        """Returns a WandB Table of embeddings generated by [model] on
        [nx_data]. To reduce data usage, [k] samples are selected randomly
        according to [args.seed].
        """
        idxs = Utils.sample(range(len(nx_data)),
            k=min(args.bs, len(nx_data)),
            seed=args.seed)
        data = Subset(Data.ZipDataset(nx_data.data, nx_data), indices=idxs)
        loader = DataLoader(data,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True)

        encoder = Utils.de_dataparallel(model).encoder
        encoder = nn.DataParallel(encoder, device_ids=args.gpus).to(device)
        
        with torch.no_grad():
            embeddings, targets = [], []
            for (x,y),(xn,_) in tqdm(loader,
                desc=f"Generating embeddings for ImageDataset [mode={mode}]",
                total=len(loader),
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout):


                if mode == "no_noise":
                    model_input = (x.to(device, non_blocking=True),)
                elif mode == "pre_fuse":
                    model_input = (xn.to(device, non_blocking=True),)

                fxn = encoder(*model_input).cpu()

                embeddings.append(fxn)
                targets.append(y)
        
        embeddings = torch.cat(embeddings, dim=0)
        targets = torch.cat(targets, dim=0).view(-1)
        return embeddings, targets
        
    @staticmethod
    def generate_images(nx_data, model, args, idxs=None, num_images=None, num_samples=None, noise_seed=None, idxs_seed=None):
        """Returns a PIL image from running [model] on [nx_data]. Each row in
        the image contains: ...
        
        Args:
        nx_data        -- ImageDataset containing images
        model           -- model to generate images with
        args            -- argparse Namespace with relevant parameters. Its
                            NUM_EVAL_IMAGES, NUM_EVAL_SAMPLES, SEED are overriden by
                            other arguments to this function when they are not None
        idxs            -- indices to [nx_data] to use
        num_images      -- number of images to use. Overriden by [idxs], overrides args.num_eval_images if specified
        num_samples     -- number of samples to generate per image, overrides args.num_eval_samples if specified
        noise_seed      -- seed used in sampling image noise if specified
        idxs_seed       -- seed used in sampling indices if specified
        """
        noise_seed = args.eval_noise_seed if noise_seed is None else noise_seed
        idxs_seed = args.eval_idxs_seed if idxs_seed is None else idxs_seed

        num_images = args.num_eval_images if num_images is None else num_images
        num_samples = args.num_eval_samples if num_samples is None else num_samples

        if idxs is None:
            idxs = Utils.sample(range(len(nx_data)), k=num_images, seed=idxs_seed)
        else:
            idxs = idxs
        
        data = Subset(Data.ZipDataset(nx_data.data, nx_data), indices=idxs)
        loader = DataLoader(data,
            batch_size=args.bs,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True)

        output = torch.zeros(len(data), num_samples+3, *nx_data[0][1].shape)

        with torch.no_grad():
            for idx,((x,_),(nx,t)) in tqdm(enumerate(loader),
                desc="Generating images from ImageDataset",
                total=len(loader),
                file=sys.stdout,
                leave=False,
                dynamic_ncols=True):
                start_idx = idx * args.bs
                stop_idx = min(len(data), (idx+1) * args.bs)

                output[start_idx:stop_idx, 0] = t
                output[start_idx:stop_idx, 1] = nx
                output[start_idx:stop_idx, 2] = model(nx).cpu()

        result = Utils.images_to_pil_image(output,
            sigmoid=(args.loss == "bce"), clip=(args.data_tr == "mnist"))
        return result

    @staticmethod
    def eval_model(nx_data, model, args, loss_fn=None):
        """Returns the loss of [model] evaluated on data in nx_data.

        Args:
        nx_data            -- ImageDataset
        model               -- dae model to evaluate
        args                -- argparse Namespace parameterizing run
        loss_fn             -- loss function to use
        use_sampled_codes   -- whether to use the codes sampled in [nx_data]
                                (dae objective) or random new ones (minimize
                                expected loss)
        """
        loss, total = 0, 0
        loss_fn = Models.get_loss_fn(args, reduction="mean")

        with torch.no_grad():
            loader = DataLoader(nx_data,
                batch_size=args.bs * 8,
                num_workers=args.num_workers,
                pin_memory=True)
            
            for xn,x in tqdm(loader,
                desc="Evaluating on ImageDataset",
                total=len(loader),
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True):

                xn = xn.to(device, non_blocking=True)
                x = x.to(device, non_blocking=True)
                loss += loss_fn(model(xn), x) * len(x)
                total += len(x)
        
        return loss.item() / total

    @staticmethod
    def get_image_dataset(model, dataset, args):
        """Returns an ImageDataset giving noised images and codes for
        [model] to use in dae training. 

        Args:
        model   -- dae model
        dataset -- ImageFolder-like dataset of non-noised images to get codes for
        args    -- argparse Namespace
        """
        with torch.no_grad():
            images = torch.zeros(len(dataset), *dataset[0][0].shape)
            noised_images = torch.zeros(len(dataset), *dataset[0][0].shape)

            loader = DataLoader(dataset,
                batch_size=args.bs,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True)

            for idx,(x,_) in tqdm(enumerate(loader),
                desc="Sampling outer loop",
                total=len(loader),
                leave=False,
                file=sys.stdout,
                dynamic_ncols=True):

                start_idx = idx * args.bs
                stop_idx = min(start_idx + args.bs, len(dataset))
                x = x.to(device, non_blocking=True)
                xn = Utils.with_noise(x, std=args.std)
                
                # Sanity check for dae. The DAE should regress to the mean
                if args.zero_half_target:
                    drop_top_idxs = torch.rand(len(x)) < .5
                    drop_bottom_idxs = ~drop_top_idxs
                    x[drop_top_idxs, :, :14, :] = 0
                    x[drop_bottom_idxs, :, 14:, :] = 0

                noised_images[start_idx:stop_idx] = xn.cpu()
                images[start_idx:stop_idx] = x.cpu()

        return ImageDataset(dataset, noised_images, images)

def get_args(args=None):
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_data_args(P)
    P = IO.parser_with_logging_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_probe_args(P)
    args, unparsed_args = P.parse_known_args() if args is None else P.parse_known_args(args)

    if args.data_tr == "mnist":
        P = IO.parser_with_mnist_training_args(argparse.ArgumentParser())
    elif args.data_tr == "cifar10":
        P = IO.parser_with_cifar_training_args(argparse.ArgumentParser())
    else:
        raise ValueError()

    args = Utils.namespace_union(args, P.parse_args(unparsed_args))

    args.uid = wandb.util.generate_id() if args.uid is None else args.uid
    args.script = "dae" if args.script is None else args.script
    args.lrs = Utils.StepScheduler.process_lrs(args.lrs)
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    if not args.probe_trials == 1:
        raise NotImplementedError(f"Running multiple probe trials is currently not supported in a script that logs to WandB.")

    return args

if __name__ == "__main__":
    args = get_args()    

    if args.resume is None:
        Utils.set_seed(args.seed)
        model = Models.get_model(args, imle=False)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=args.wd)
        last_epoch = -1
    else:
        states = torch.load(args.resume)
        Utils.set_seed(states["seeds"])
        args = argparse.Namespace(**vars(states["args"]) | vars(args))
        model = Models.get_model(args, imle=True)
        model.load_state_dict(states["model"], strict=False)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=args.wd)
        optimizer.load_state_dict(states["optimizer"])
        model = model.to(device)
        last_epoch = states["epoch"]
        args.uid = states["args"].uid if args.continue_run else args.uid

    scheduler = Utils.StepScheduler(optimizer, args.lrs, last_epoch=last_epoch)
    loss_fn = Models.get_loss_fn(args, reduction="mean")
    data_tr, data_val = Data.get_data_from_args(args)

    tqdm.write(f"---ARGS---\n{Utils.sorted_namespace(args)}\n----------")
    tqdm.write(f"---MODEL---\n{model.module}")
    tqdm.write(f"---OPTIMIZER---\n{optimizer}")
    tqdm.write(f"---SCHEDULER---\n{scheduler}")
    tqdm.write(f"---TRAINING DATA---\n{data_tr}")

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL", entity="apex-lab",
        name=os.path.basename(dae_model_folder(args)),
        resume="allow" if args.continue_run else "never",
        settings=wandb.Settings(code_dir=os.path.dirname(__file__)))

    if not args.save_iter == 0:
        _ = Utils.save_code_under_folder(dae_model_folder(args))

    cur_step = (last_epoch + 1) * math.ceil(len(data_tr) / args.bs)
    num_steps = args.epochs * math.ceil(len(data_tr) / args.bs)
    _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step)
    for epoch in tqdm(range(last_epoch + 1, args.epochs),
        dynamic_ncols=True,
        desc="Epochs",
        file=sys.stdout):

        epoch_dataset = ImageDataset.get_image_dataset(
            model=model,
            dataset=data_tr,
            args=args)
        loader = DataLoader(epoch_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=args.bs,
            num_workers=args.num_workers)
       
        for xn,x in tqdm(loader,
            desc="Batches",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout):

            xn = xn.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

            loss = loss_fn(model(xn), x)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            cur_step += 1
        
        if epoch % args.eval_iter == 0 or epoch == args.epochs - 1:
            _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step,
                nx_data_tr=epoch_dataset)

        if ((not args.save_iter == 0 and epoch % args.save_iter == 0)
            or epoch in args.save_epochs or epoch == args.epochs -1):
            _ = Utils.save_state(model, optimizer,
                args=args,
                epoch=epoch,
                folder=dae_model_folder(args))
        elif args.save_iter == -1 or ():
            raise NotImplementedError()
        elif args.save_iter == -2 and time.time() - wandb.run.start_time > 1800:
            _ = Utils.save_state(model, optimizer,
                args=args,
                epoch=epoch,
                folder=dae_model_folder(args),
                delete_prior_state=True)

        scheduler.step(epoch)
        
