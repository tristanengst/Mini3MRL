import argparse
from collections import OrderedDict, defaultdict
import numpy as np
import os
import plotly.express as px
from PIL import Image
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io

from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_descriptor")

def conditional_make_folder(f):
    try:
        os.makedirs(f)
    except:
        pass

def matrix_to_stats(m, matrix_name=""):
    """Returns a dictionary of statistics for matrix [m] for logging to WandB."""
    matrix_name = f"{matrix_name}_" if len(matrix_name) > 0 else matrix_name
    m = m.squeeze()
    if not len(m.shape) == 2:
        tqdm.write(f"MATRIX_TO_STATS: Matrix had {len(m.shape)} dimensions after squeezing out singular ones. Singular values will not be logged.")
        two_d_stats = {}
    else:
        singular_vals = torch.linalg.svdvals(m)
        two_d_stats = {
            f"weights/{matrix_name}singular_vals": singular_vals,
            f"weights/{matrix_name}singular_vals_std": torch.mean(singular_vals),
            f"weights/{matrix_name}singular_vals_mean": torch.std(singular_vals),
        }
    
    return two_d_stats | {
        f"weights/{matrix_name}mean": torch.mean(m),
        f"weights/{matrix_name}std": torch.std(m),
        f"weights/{matrix_name}": m,
        f"weights/{matrix_name}norm": torch.linalg.norm(m) / (m.view(-1).shape[0] ** .5),
    }

def with_noise(x, std=.8, seed=None):
    if seed is None:
        return x + torch.randn(*x.shape, device=x.device) * std
    else:
        noise = torch.zeros(*x.shape, device=x.device)
        noise.normal_(generator=torch.Generator(x.device).manual_seed(seed))
        return x + noise * std

def save_state(model, optimizer, args, epoch, folder):
    """Saves [model], [optimizer], [args], and [epoch] along with Python, NumPy,
    and PyTorch random seeds to 'folder/epoch.pt'.

    Args:
    model       -- model to be saved
    optimizer   -- optimizer to be saved
    args        -- argparse Namespace used to create run
    epoch       -- epoch number to save with
    folder      -- folder inside which to save everything
    """
    state_dict = {"model": de_dataparallel(model).cpu().state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "seeds": {"random_seed": random.getstate(),
            "torch_seed": torch.get_rng_state(),
            "torch_cuda_seed": torch.cuda.get_rng_state(),
            "numpy_seed": np.random.get_state()
        }
    }
    _ = conditional_make_folder(folder)
    torch.save(state_dict, f"{folder}/{epoch}.pt")
    model.to(torch.device("cuda") if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        tqdm.write(f"Set the NumPy, PyTorch, and Random modules seeds to {seed}")
    elif isinstance(seed, dict):
        random.setstate(seed["random_seed"])
        np.random.set_state(seed["numpy_seed"])
        torch.set_rng_state(seed["torch_seed"])
        torch.cuda.set_rng_state(seed["torch_cuda_seed"])
        tqdm.write(f"Reseeded program with old seed")
    else:
        raise ValueError(f"Seed should be int or contain resuming keys")

    return seed

def sample(select_from, k=-1, seed=0):
    """Returns [k] items sampled without replacement from [select_from] with
    seed [seed], without changing the internal seed of the program. This
    explicitly ensures reproducability.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        result = random.sample(select_from, k=k)
    except ValueError as e:
        tqdm.write(f"Tried to sample {k} from {len(select_from)} things")
        raise e
    random.setstate(state)
    return result

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    if isinstance(xs, list) or isinstance(xs, set) or isinstance(xs, tuple):
        result = []
        for x in xs:
            result += flatten(x)
        return result
    else:
        return [xs]

def images_to_pil_image(images):
    """Returns tensor datastructure [images] as a PIL image that can thus be
    easily saved.

    This function should handle a myriad of inputs and thus pull complexity from
    other places in the code inside of it.
    """
    if len(images.shape) == 3:
        images = images.view(1, 1, *images.shape)
    elif len(images.shape) == 4: # Assume batch index should be rows
        images = images.view(images.shape[0], 1, *images.shape[1:])
    elif len(images.shape) == 5:
        images = images

    fig, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images),
        squeeze=False)

    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            image = torch.clip((image * 255), 0, 255).int().cpu()
            image = image.permute(-2, -1, 0) if len(image.shape) == 3 else image
            axs[i, j].imshow(np.asarray(image), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=256)
    buf.seek(0)
    plt.close("all")
    result = Image.open(buf)
    del buf
    return result

def embeddings_to_pil_image(embeddings, classes, method="plain"):
    """Returns a PIL image of [embeddings] and [classes] represented in feature
    space.

    Args:
    embeddings  -- NxD tensor of model embeddings
    classes     -- N-dimensional tensor of classes
    method      -- method by which to represent the data
    """
    n,d = embeddings.shape
    embeddings = embeddings.cpu().numpy()
    classes = [str(c) for c in classes.cpu().numpy().tolist()]
    if d == 2 and method == "plain":
        fig = px.scatter(x=embeddings[:, 0], y=embeddings[:, 1], color=classes)
        return Image.open(io.BytesIO(fig.to_image(format="png")))
    else:
        tqdm.write(f"Not implemented for higher dimensional feature spaces, outputting image of zeros instead")
        return Image.fromarray(np.zeros(shape=(128,128), dtype=np.int8))

def de_dataparallel(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def sorted_namespace(args):
    """Returns argparse Namespace [args] after sorting the args in it by key
    value. The utility of this is printing.
    """
    d = vars(args)
    return argparse.Namespace(**{k: d[k] for k in sorted(d.keys())})

class StepScheduler:
    """StepLR but with easier control.
    
    Args:
    optimizer   -- optimizer to step
    lrs         -- list where sequential pairs of elements describe a step index
                    and the learning rate for that step and subsequent steps
                    until a new learning rate is specified
    last_epoch  -- the last run step
    """
    def __init__(self, optimizer, lrs, last_epoch=-1):
        super(StepScheduler, self).__init__()
        self.optimizer = optimizer
        
        # Get a mapping from epoch indices to the learning rates they should if
        # the learning rate should change at the start of the epoch
        keys = [lrs[idx] for idx in range(0, len(lrs) -1, 2)]
        vals = [lrs[idx] for idx in range(1, len(lrs), 2)]
        self.schedule = OrderedDict(list(zip(keys, vals)))

        # Create a dictionary that implements (a) a fast mapping from steps to
        # the learning rate they should have, and (b) support for infinite steps
        # using the last learning rate
        self.step2lr = defaultdict(lambda: self.schedule[max(self.schedule.keys())])
        self.step2lr[-1] = self.schedule[0]
        cur_lr = self.schedule[0]
        for s in range(max(self.schedule.keys())):
            if s in self.schedule:
                cur_lr = self.schedule[s]
            self.step2lr[s] = cur_lr

        self.cur_step = last_epoch    
        self.step()
    
    def __str__(self): return f"{self.__class__.__name__} [schedule={dict(self.schedule)} cur_step={self.cur_step} lr={self.get_lr()}]"

    def get_lr(self): return self.step2lr[self.cur_step]
        
    def step(self, cur_step=None):
        cur_step = self.cur_step if cur_step is None else cur_step

        for pg in self.optimizer.param_groups:
            pg["lr"] = self.step2lr[cur_step]

        self.cur_step = cur_step + 1

    @staticmethod
    def process_lrs(lrs):
        """Returns a list where even elements give a step index and are integers
        and odd elements give the float learning rate starting at the prior even
        element step.

        This is intended to be run on the initial float-valied LRS attribute
        collected through argparse, and will raise argparse errors if the LRS
        specification is bad.
        """
        lrs = [float(l) for l in lrs]
        def is_increasing(l): return sorted(l) == l and len(l) == len(set(l))

        if not len(lrs) % 2 == 0:
            raise argparse.ArgumentTypeError(f"--lrs must have an even number of values")
        if not is_increasing([l for idx,l in enumerate(lrs) if idx % 2 == 0]):
            raise argparse.ArgumentTypeError(f"--lrs must have strictly increasing keys (even values)")
        if not lrs[0] == int(0):
            raise argparse.ArgumentTypeError(f"--lrs should begin with 0")
        else:
            return [int(l) if idx % 2 == 0 else float(l)
                for idx,l in enumerate(lrs)]