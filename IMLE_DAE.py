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

import Models
import IO
import LinearProbe
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def imle_model_folder(args, make_folder=True):
    subsample = "_all" if args.subsample is None else args.subsample
    suffix = "" if args.suffix is None else f"-{arg.suffix}"
    folder = f"{args.save_folder}/imle_models/imle-bs{args.bs}-epochs{args.epochs}-ipe{args.ipe}-lr{args.bs}-ns{args.ns}-std{args.std}-subsample{subsample}seed{args.seed}-{args.uid}{suffix}"

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
    tqdm.write(f"Epoch {epoch:5}/{args.epochs} step {args.ipe * len(loader_tr) * epoch}/{len(loader_tr) * args.ipe * args.epochs}- lr={scheduler.get_last_lr()[0]:.5e} loss/tr={loss_tr:.5f} loss/te={loss_te:.5f} acc/te={acc_te:.5f}")

    # Create an image to visualize how the model is doing
    image_save_folder = f"{imle_model_folder(args, make_folder=True)}/images"
    Utils.conditional_make_folder(image_save_folder)
    image_path = f"{image_save_folder}/{epoch}_te.png"
    with torch.no_grad():
        fxn_te = model(nx_te[:8], num_z=6, seed=args.seed).view(8, 6, 784)
    image = torch.cat([x_te[:8].unsqueeze(1), nx_te[:8].unsqueeze(1), fxn_te], dim=1)
    Utils.images_to_pil_image(image).save(image_path)

    wandb.log({
        "acc/te": acc_te,
        "loss/te": loss_te,
        "loss/tr": loss_tr,
        "lr": scheduler.get_last_lr()[0],
        "train_step": epoch * len(loader_tr) * args.ipe,
        "images/te": wandb.Image(image_path)
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
        x       -- tensor of non-noised data to get codes for
        args    -- argparse Namespace
        """
        with torch.no_grad():
            least_losses = torch.ones(len(dataset), device=device) * float("inf")
            best_latents = torch.zeros(len(dataset), 512, device=device)
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
                x = x.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                xn = Utils.with_noise(x, std=args.std)

                for sample_idx in tqdm(range(args.ns),
                    desc="Sampling inner loop",
                    leave=False,
                    dynamic_ncols=True):

                    z = model.get_codes(len(xn), device=xn.device)
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

def get_args():
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_training_args(P)
    P = IO.parser_with_probe_args(P)
    P = IO.parser_with_imle_args(P)

    args = P.parse_args()
    args.uid = wandb.util.generate_id()
    return args

if __name__ == "__main__":
    args = get_args()
    tqdm.write(str(args))
    Utils.set_seed(args.seed)

    wandb.init(anonymous="allow", id=args.uid, config=args,
        mode=args.wandb, project="Mini3MRL",
        name=os.path.basename(imle_model_folder(args)))

    tqdm.write(f"Will save to {imle_model_folder(args)}")

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
        num_workers=24,
        persistent_workers=True,
        pin_memory=True)
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=24,
        persistent_workers=True,
        prefetch_factor=8,
        pin_memory=True)

    model = Models.IMLE_DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        step_size=args.epochs // 5,
        gamma=.3)

    _ = evaluate(model, loader_tr, loader_te, scheduler, args, epoch=0)
    for epoch in tqdm(range(args.epochs),
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
        
        if epoch % args.eval_iter == 0 or epoch == args.epochs - 1:
            _ = evaluate(model, loader_tr, loader_te, scheduler, args, epoch+1)
    
        scheduler.step()
    
        



        
