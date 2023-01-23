import argparse
import os
import itertools
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

def imle_model_folder(args, make_folder=False):
    data_str = Data.dataset_pretty_name(args.data_tr)
    suffix = "" if args.suffix is None else f"-{args.suffix}"
    job_id = "" if args.job_id is None else f"-{args.job_id}"
    folder = f"{args.save_folder}/models_{args.script}/{data_str}-bs{args.bs}-epochs{args.epochs}-ipe{args.ipe}-lr{args.bs}-ns{args.ns}-nshot{args.n_way}-nway{args.n_shot}-std{args.std}-seed{args.seed}-{args.uid}{job_id}{suffix}"

    if make_folder:
        Utils.conditional_make_folder(folder)

    return folder

def evaluate(model, loader_tr, loader_val, scheduler, args, cur_step):
    eval_loss_fn = nn.MSELoss(reduction="sum")
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
            nx_val = Utils.with_noise(x_val, std=args.std, seed=args.seed)
            fxn_val = model(nx_val)
            loss_val += eval_loss_fn(fxn_val, x_val)
            total_val += len(x_val)
    
    loss_tr = loss_tr.item() / total_tr
    loss_val = loss_val.item() / total_val

    acc_val = LinearProbe.linear_probe(model, loader_tr, loader_val, args)
    tqdm.write(f"Step {cur_step}/{len(loader_tr) * args.ipe * args.epochs} - lr={scheduler.get_lr():.5e} loss/tr={loss_tr:.5f} loss/te={loss_val:.5f} acc/te={acc_val:.5f}")


    image_shape = x_val.shape[1:]

    # Create an image to visualize how the model is doing
    image_save_folder = f"{imle_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    image_path_tr = f"{image_save_folder}/{cur_step}_tr.png"
    with torch.no_grad():
        fxn_tr = model(nx_tr[:8], num_z=6, seed=args.seed).view(8, 6, *image_shape)
    image = torch.cat([x_tr[:8].unsqueeze(1), nx_tr[:8].unsqueeze(1), fxn_tr], dim=1)
    Utils.images_to_pil_image(image).save(image_path_tr)

    # Create an image to visualize how the model is doing
    Utils.conditional_make_folder(image_save_folder)
    image_path_val = f"{image_save_folder}/{cur_step}_val.png"
    with torch.no_grad():
        fxn_val = model(nx_val[:8], num_z=6, seed=args.seed).view(8, 6, *image_shape)
    image = torch.cat([x_val[:8].unsqueeze(1), nx_val[:8].unsqueeze(1), fxn_val], dim=1)
    Utils.images_to_pil_image(image).save(image_path_val)

    wandb.log({
        "acc/te": acc_val,
        "loss/te": loss_val,
        "loss/tr": loss_tr,
        "lr": scheduler.get_lr(),
        "train_step": cur_step,
        "images/te": wandb.Image(image_path_val),
        "images/tr": wandb.Image(image_path_tr)
    })

class ImageLatentDataset(Dataset):

    def __init__(self, noised_images, latents, images):
        super(ImageLatentDataset, self).__init__()
        self.noised_images = noised_images.cpu()
        self.latents = latents.cpu()
        self.images = images.cpu()
    
    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        return self.noised_images[idx], self.latents[idx], self.images[idx]

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
                shuffle=False,
                pin_memory=True)

            for idx,(x,_) in tqdm(enumerate(loader),
                desc="Sampling outer loop",
                total=len(loader),
                leave=False,
                dynamic_ncols=True):

                start_idx = idx * args.code_bs
                stop_idx = min(start_idx + args.code_bs, len(dataset))
                x = x.to(device, non_blocking=True).view(x.shape[0], -1)
                xn = Utils.with_noise(x, std=args.std)

                for sample_idx in tqdm(range(args.ns),
                    desc="Sampling inner loop",
                    leave=False,
                    dynamic_ncols=True):

                    z = Utils.de_dataparallel(model).get_codes(len(xn), device=xn.device)
                    fxn = model(xn, z)
                    losses = loss_fn(fxn, x)
                    losses = torch.sum(losses, dim=1)

                    change_idxs = (losses < least_losses[start_idx:stop_idx])
                    least_losses[start_idx:stop_idx][change_idxs] = losses[change_idxs]
                    best_latents[start_idx:stop_idx][change_idxs] = z[change_idxs]

                noised_images.append(xn.cpu())
                images.append(x.cpu())

        noised_images = torch.cat(noised_images, dim=0)
        images = torch.cat(images, dim=0)
        return ImageLatentDataset(noised_images, best_latents.cpu(), images)

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
    return args

if __name__ == "__main__":
    args = get_args()

    if args.resume is None:
        model = Models.get_model(args, imle=True)
        last_epoch = -1
    else:
        states = torch.load(args.resume)
        args = argparse.Namespace(**vars(states["args"]) | vars(args))
        model = Models.get_model(args, imle=True)
        model.load_state_dict(states["model"])
        model = model.to(device)
        last_epoch = states["epoch"]

    Utils.set_seed(args.seed)
    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL", entity="apex-lab",
        name=os.path.basename(imle_model_folder(args)))

    # We probably want to discard the old optimizer state
    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1, # Reset by the scheduler
        weight_decay=1e-5)
    scheduler = Utils.StepScheduler(optimizer, args.lrs)
    loss_fn = nn.BCELoss()

    # Get DataLoaders over the training and validation data. In this context,
    # validation data is data we use for validating the linear probe, ie. the
    # linear probe and model are trained on the training data.
    if args.data_val is None:
        data_tr = Data.get_dataset(args.data_tr, split="train", transform=Data.get_transforms_tr(args))
        data_tr = Data.get_fewshot_dataset(data_tr, n_way=args.n_way, n_shot=args.n_shot, seed=args.seed)
        data_val = Data.ImageFolderSubset.complement(data_tr, replace_transform=Data.get_transforms_te(args))
    else:
        data_tr = Data.get_dataset(args.data_tr, split="train", transform=Data.get_transforms_tr(args))
        data_tr = Data.get_fewshot_dataset(data_tr, n_way=args.n_way, n_shot=args.n_shot, seed=args.seed)
        data_val = Data.get_dataset(args.data_val, split="test", transform=Data.get_transforms_te(args))
        data_val = Data.get_fewshot_dataset(data_val, n_way=args.n_way, n_shot=args.n_shot, seed=args.seed)
        
    tqdm.write(f"TRAINING DATA\n{data_tr}")
    tqdm.write(f"VALIDATION DATA\n{data_val}")
    
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True)
    
    tqdm.write(f"----------\n{args}\n----------")
    tqdm.write(f"Will save to {imle_model_folder(args)}")

    cur_step = (last_epoch + 1) * args.ipe * len(data_tr) // args.bs
    num_steps = args.ipe * len(data_tr) // args.bs
    _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
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
        
        if epoch % args.eval_iter == 0 or epoch == args.epochs - 1:
            _ = evaluate(model, loader_tr, loader_val, scheduler, args, cur_step)
        
        if not args.save_iter == 0 and epoch % args.save_iter == 0:
            state_dict = {"model": Utils.de_dataparallel(model).cpu().state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(state_dict, imle_model_folder(args, make_folder=True))
            model.to(device)
        elif args.save_iter == -1:
            raise NotImplementedError()
    
        scheduler.step(epoch)
    
        



        
