import argparse
import itertools
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
                xn = Utils.with_noise(x)

                for sample_idx in tqdm(range(args.ns),
                    desc="Sampling inner loop",
                    leave=False,
                    dynamic_ncols=True):

                    z = model.get_codes(len(xn), device=xn.device)
                    losses = torch.sum(loss_fn(model(xn, z), x), dim=1)

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

    model = Models.IMLE_DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.BCELoss()

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
        chain_loader = itertools.chain(*[loader] * args.steps_per_image)

        for xn,z,x in tqdm(chain_loader,
            desc="Batches",
            total=len(loader) * args.steps_per_image,
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

            eval_loss_fn = nn.MSELoss(reduction="sum")
            loss_tr, loss_te = 0, 0
            total_tr, total_te = 0, 0
            for x,_ in loader_tr:
                x = x.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                nx = Utils.with_noise(x)
                fxn = model(nx)
                loss_tr += eval_loss_fn(fxn, x)
                total_tr += len(x)

            for x,_ in loader_te:
                x = x.to(device, non_blocking=True)
                x = x.view(x.shape[0], -1)
                nx = Utils.with_noise(x)
                fxn = model(nx)
                loss_te += eval_loss_fn(fxn, x)
                total_te += len(x)
            
            loss_tr = loss_tr.item() / total_tr
            loss_te = loss_te.item() / total_te

            acc_te = LinearProbe.linear_probe(model, loader_tr, loader_te, args)
            tqdm.write(f"Epoch {epoch+1:5}/{args.epochs} - loss/tr={loss_tr:.5f} loss/te={loss_te:.5f} acc/te={acc_te:.5f}")
    
        



        
