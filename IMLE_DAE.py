import argparse
import os
import itertools
import numpy as np
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

def imle_model_folder(args, make_folder=False):
    data_str = Data.dataset_pretty_name(args.data_tr)
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    job_id = "" if args.job_id is None else f"-{args.job_id}"
    lrs = "_".join([f"{lr:.2e}" for idx,lr in enumerate(args.lrs) if idx % 2 == 1])
    folder = f"{args.save_folder}/models_{args.script}/{args.script}-{data_str}-bs{args.bs}-epochs{args.epochs}-ipe{args.ipe}-lr{lrs}-ns{args.ns}-nshot{args.n_way}-nway{args.n_shot}-seed{args.seed}-{args.uid}{job_id}{suffix}"

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
            loss_fn=nn.BCELoss(reduction="none"),
            dataset=data_tr,
            args=args)
    nxz_data_val = ImageLatentDataset.get_image_latent_dataset(
        model=model,
        loss_fn=nn.BCELoss(reduction="none"),
        dataset=data_val,
        args=args)

    # Generate embeddings.
    embeds_post_fuse_tr, targets_post_fuse_tr = ImageLatentDataset.generate_embeddings(nxz_data_tr, model, args, mode="post_fuse")
    embeds_pre_fuse_tr, targets_pre_fuse_tr = ImageLatentDataset.generate_embeddings(nxz_data_tr, model, args, mode="pre_fuse")
    embeds_post_fuse_val, targets_post_fuse_val = ImageLatentDataset.generate_embeddings(nxz_data_val, model, args, mode="post_fuse")
    embeds_pre_fuse_val, targets_pre_fuse_val = ImageLatentDataset.generate_embeddings(nxz_data_val, model, args, mode="pre_fuse")
    embeds_no_noise_tr, targets_no_noise_tr = ImageLatentDataset.generate_embeddings(nxz_data_tr, model, args, mode="no_noise")
    embeds_no_noise_val, targets_no_noise_val = ImageLatentDataset.generate_embeddings(nxz_data_val, model, args, mode="no_noise")

    embedding_results = {
        "embeds/post_fuse_vis/tr": wandb.Image(Utils.embeddings_to_pil_image(embeds_post_fuse_tr, targets_post_fuse_tr)),
        "embeds/post_fuse_vis/val": wandb.Image(Utils.embeddings_to_pil_image(embeds_post_fuse_val, targets_post_fuse_val)),
        "embeds/pre_fuse_vis/tr": wandb.Image(Utils.embeddings_to_pil_image(embeds_pre_fuse_tr, targets_pre_fuse_tr)),
        "embeds/pre_fuse_vis/val": wandb.Image(Utils.embeddings_to_pil_image(embeds_pre_fuse_val, targets_pre_fuse_val)),
        "embeds/no_noise_vis/tr": wandb.Image(Utils.embeddings_to_pil_image(embeds_no_noise_tr, targets_no_noise_tr)),
        "embeds/no_noise_vis/val": wandb.Image(Utils.embeddings_to_pil_image(embeds_no_noise_val, targets_no_noise_val)),

        "embeds/post_fuse_mean/tr": torch.mean(embeds_post_fuse_tr),
        "embeds/post_fuse_mean/val": torch.mean(embeds_post_fuse_val),
        "embeds/pre_fuse_mean/tr": torch.mean(embeds_pre_fuse_tr),
        "embeds/pre_fuse_mean/val": torch.mean(embeds_pre_fuse_val),
        "embeds/no_noise_mean/tr": torch.mean(embeds_no_noise_tr),
        "embeds/no_noise_mean/val": torch.mean(embeds_no_noise_val),

        "embeds/post_fuse_feat_std/tr": torch.mean(torch.std(embeds_post_fuse_tr, dim=0)),
        "embeds/post_fuse_feat_std/val": torch.mean(torch.std(embeds_post_fuse_val, dim=0)),
        "embeds/pre_fuse_feat_std/tr": torch.mean(torch.std(embeds_pre_fuse_tr, dim=0)),
        "embeds/pre_fuse_feat_std/val": torch.mean(torch.std(embeds_pre_fuse_val, dim=0)),
        "embeds/no_noise_feat_std/tr": torch.mean(torch.std(embeds_no_noise_tr, dim=0)),
        "embeds/no_noise_feat_std/val": torch.mean(torch.std(embeds_no_noise_val, dim=0)),

        "embeds/post_fuse_ex_std/tr": torch.mean(torch.std(embeds_post_fuse_tr, dim=1)),
        "embeds/post_fuse_ex_std/val": torch.mean(torch.std(embeds_post_fuse_val, dim=1)),
        "embeds/pre_fuse_ex_std/tr": torch.mean(torch.std(embeds_pre_fuse_tr, dim=1)),
        "embeds/pre_fuse_ex_std/val": torch.mean(torch.std(embeds_pre_fuse_val, dim=1)),
        "embeds/no_noise_ex_std/tr": torch.mean(torch.std(embeds_no_noise_tr, dim=1)),
        "embeds/no_noise_ex_std/val": torch.mean(torch.std(embeds_no_noise_val, dim=1)),

        "embeds/post_fuse_abs/tr": torch.mean(torch.abs(embeds_post_fuse_tr)),
        "embeds/post_fuse_abs/val": torch.mean(torch.abs(embeds_post_fuse_val)),
        "embeds/pre_fuse_abs/tr": torch.mean(torch.abs(embeds_pre_fuse_tr)),
        "embeds/pre_fuse_abs/val": torch.mean(torch.abs(embeds_pre_fuse_val)),
        "embeds/no_noise_abs/tr": torch.mean(torch.abs(embeds_no_noise_tr)),
        "embeds/no_noise_abs/val": torch.mean(torch.abs(embeds_no_noise_val)),
    }
    
    # Generate images
    images_tr = ImageLatentDataset.generate_images(nxz_data_tr, model, args)
    images_val = ImageLatentDataset.generate_images(nxz_data_val, model, args)
    if args.save_iter > 0:
        image_save_folder = f"{imle_model_folder(args, make_folder=True)}/images"
        Utils.conditional_make_folder(image_save_folder)
        images_tr.save(f"{image_save_folder}/{cur_step}_tr.png")
        images_val.save(f"{image_save_folder}/{cur_step}_val.png")

    # Evaluate on the proxy task            
    loss_tr_min = ImageLatentDataset.eval_model(nxz_data_tr, model, args, use_sampled_codes=True)
    loss_tr_mean = ImageLatentDataset.eval_model(nxz_data_tr, model, args, use_sampled_codes=False)
    loss_val_min = ImageLatentDataset.eval_model(nxz_data_val, model, args, use_sampled_codes=True)
    loss_val_mean = ImageLatentDataset.eval_model(nxz_data_val, model, args, use_sampled_codes=False)

    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True)

    tqdm.write(f"Step {cur_step}/{len(loader_tr) * args.ipe * args.epochs} - lr={scheduler.get_lr():.5e} loss/min/tr={loss_tr_min:.5f} loss/min/val={loss_val_min:.5f} loss/mean/tr={loss_tr_mean:.5f} loss/mean/val={loss_val_mean:.5f}")

    # Evaluate on the probing task
    epoch = cur_step / (len(loader_tr) * args.ipe)
    if epoch % args.probe_iter == 0 or epoch == args.epochs - 1:
        probe_results = LinearProbe.probe(model, loader_tr, loader_val, args)
    else:
        probe_results = {}

    wandb.log(probe_results | embedding_results | {
        "loss/min/tr": loss_tr_min,
        "loss/min/val": loss_val_min,
        "loss/mean/tr": loss_tr_mean,
        "loss/mean/val": loss_val_mean,
        "lr": scheduler.get_lr(),
        "train_step": cur_step,
        "images/val": wandb.Image(images_val),
        "images/tr": wandb.Image(images_tr),
        "epoch": cur_step // (len(loader_tr) * args.ipe)
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
    def generate_embeddings(nxz_data, model, args, mode="post_fuse", use_sampled_codes=True, k=1000):
        """Returns a WandB Table of embeddings generated by [model] on
        [nxz_data]. To reduce data usage, [k] samples are selected randomly
        according to [args.seed].

        Args:
        post_fuse   -- get embeddings after or before fusing them with codes
        """
        idxs = Utils.sample(range(len(nxz_data)), k=k, seed=args.seed)
        data = Subset(Data.ZipDataset(nxz_data.data, nxz_data), indices=idxs)
        loader = DataLoader(data,
            batch_size=args.code_bs,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True)

        if mode == "post_fuse":
            encoder = Utils.de_dataparallel(model).to_encoder_with_ada_in(
                use_mean_representation=False)
        else:
            encoder = Utils.de_dataparallel(model).encoder
        encoder = nn.DataParallel(encoder, device_ids=args.gpus).to(device)
        
        with torch.no_grad():
            embeddings, targets = [], []
            for (x,y),(xn,z,_) in tqdm(loader,
                desc=f"Generating embeddings for ImageLatentDataset [mode={mode}]",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                if mode == "no_noise":
                    model_input = (x.to(device, non_blocking=True),)
                elif mode == "pre_fuse":
                    model_input = (xn.to(device, non_blocking=True),)
                elif mode == "post_fuse":
                    z = z.to(device, non_blocking=True) if use_sampled_codes else None
                    model_input = (xn.to(device, non_blocking=True), z)
                    
                fxn = encoder(*model_input).cpu()

                embeddings.append(fxn)
                targets.append(y)
        
        embeddings = torch.cat(embeddings, dim=0)
        targets = torch.cat(targets, dim=0).view(-1)
        return embeddings, targets
        
    @staticmethod
    def generate_images(nxz_data, model, args, idxs=None, num_images=None, num_samples=None, seed=None):
        """Returns a PIL image from running [model] on [nxz_data].
        
        Args:
        nxz_data    -- ImageLatentDataset containing images
        model       -- model to generate images with
        args        -- argparse Namespace with relevant parameters. Its
                        NUM_EVAL_IMAGES, NUM_EVAL_SAMPLES, SEED are overriden by
                        other arguments to this function when they are not None
        idxs        -- indices to [image_latent_data] to use
        num_images  -- number of images to use. Overriden by [idxs]
        num_samples -- number of samples to generate per image
        seed        -- seed to use for latent codes 
        """
        if idxs is None and num_images is None:
            idxs = Utils.sample(range(len(nxz_data)), k=args.num_eval_images)
        elif idxs is None and not num_images is None:
            idxs = Utils.sample(range(len(nxz_data)), k=num_images)
        else:
            idxs = idxs
        
        num_samples = args.num_eval_samples if num_samples is None else num_samples

        data = Subset(Data.ZipDataset(nxz_data.data, nxz_data), indices=idxs)
        loader = DataLoader(data,
            batch_size=max(1, args.code_bs // num_samples),
            num_workers=args.num_workers,
            pin_memory=True)

        image_shape = data[0][0][0].shape

        with torch.no_grad():
            inputs, noised_inputs, generates = [], [], []
            for (x,_),(nx,_,_) in tqdm(loader,
                desc="Generating images from ImageLatentDataset",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                fx = model(nx, num_z=num_samples, seed=args.seed).cpu()
                inputs.append(x)
                noised_inputs.append(nx)
                generates.append(fx)

        inputs = torch.cat(inputs, dim=0).unsqueeze(1)
        noised_inputs = torch.cat(noised_inputs, dim=0).unsqueeze(1)
        generates = torch.cat(generates, dim=0).view(len(idxs), num_samples, *image_shape)

        images = torch.cat([inputs, noised_inputs, generates], dim=1)
        images = images.view(len(idxs), num_samples + 2, *image_shape)
        return Utils.images_to_pil_image(images)

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
        loss_fn = nn.BCELoss(reduction="mean") if loss_fn is None else loss_fn
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
        dataset -- dataset of non-noised images to get codes for
        args    -- argparse Namespace
        """
        with torch.no_grad():
            least_losses = torch.ones(len(dataset), device=device) * float("inf")
            best_latents = Utils.de_dataparallel(model).get_codes(len(dataset), device=device)
            images = []
            noised_images = []

            loader = DataLoader(dataset,
                batch_size=args.code_bs,
                num_workers=args.num_workers,
                shuffle=False,
                pin_memory=True)

            for idx,(x,_) in tqdm(enumerate(loader),
                desc="Sampling outer loop",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                start_idx = idx * args.code_bs
                stop_idx = min(start_idx + args.code_bs, len(dataset))
                x = x.to(device, non_blocking=True)
                xn = Utils.with_noise(x, std=args.std)

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

                noised_images.append(xn.cpu())
                images.append(x.cpu())

        noised_images = torch.cat(noised_images, dim=0)
        images = torch.cat(images, dim=0)
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
    args.script = "imle" if args.script is None else args.script
    args.lrs = Utils.StepScheduler.process_lrs(args.lrs)
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    assert args.probe_iter % args.eval_iter == 0

    return args

if __name__ == "__main__":
    args = get_args()    

    if args.resume is None:
        Utils.set_seed(args.seed)
        model = Models.get_model(args, imle=True)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=1e-5)
        last_epoch = -1
    else:
        states = torch.load(args.resume)
        Utils.set_seed(states["seeds"])
        args = argparse.Namespace(**vars(states["args"]) | vars(args))
        model = Models.get_model(args, imle=True)
        model.load_state_dict(states["model"], strict=False)
        model = nn.DataParallel(model, device_ids=args.gpus).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1,
            weight_decay=1e-5)
        optimizer.load_state_dict(states["optimizer"])
        model = model.to(device)
        last_epoch = states["epoch"]

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL", entity="apex-lab",
        name=os.path.basename(imle_model_folder(args)))
    
    scheduler = Utils.StepScheduler(optimizer, args.lrs)
    loss_fn = nn.BCELoss()

    data_tr, data_val = Data.get_data_from_args(args)
        
    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_val}")
    
    tqdm.write(f"-------\n{Utils.sorted_namespace(args)}\n-------")
    tqdm.write(f"Will save to {imle_model_folder(args)}")

    cur_step = (last_epoch + 1) * args.ipe * len(data_tr) // args.bs
    num_steps = args.ipe * len(data_tr) // args.bs
    _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step)
    for epoch in tqdm(range(last_epoch + 1, args.epochs),
        dynamic_ncols=True,
        desc="Epochs"):

        epoch_dataset = ImageLatentDataset.get_image_latent_dataset(
            model=model,
            loss_fn=nn.BCELoss(reduction="none"),
            dataset=data_tr,
            args=args)
        loader = DataLoader(epoch_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=args.bs,
            persistent_workers=True,
            num_workers=args.num_workers)
        chain_loader = itertools.chain(*[loader] * args.ipe)
        chain_loader_len = len(loader) * args.ipe

        for idx,(xn,z,x) in tqdm(enumerate(chain_loader),
            desc="Batches",
            total=len(loader) * args.ipe,
            leave=False,
            dynamic_ncols=True):

            xn = xn.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)

            fxn = model(xn, z) 
            loss = loss_fn(fxn, x)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            cur_step += 1

        # Otherwise the worker threads hang around and cause problems?
        del loader, chain_loader
        
        if epoch % args.eval_iter == 0 or epoch == args.eval_iter - 1:
            _ = evaluate(model, data_tr, data_val, scheduler, args, cur_step,
                nxz_data_tr=epoch_dataset)
        
        if ((not args.save_iter == 0 and epoch % args.save_iter == 0)
            or epoch in args.save_epochs):
            _ = Utils.save_state(model, optimizer,
                args=args,
                epoch=epoch,
                folder=imle_model_folder(args))
        elif args.save_iter == -1:
            raise NotImplementedError()
    
        scheduler.step(epoch)
        
