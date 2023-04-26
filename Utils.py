import argparse
from collections import OrderedDict, defaultdict
import numpy as np
import os
import plotly.express as px
from PIL import Image
import random
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import io

from tqdm import tqdm

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_descriptor")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def with_noise(x, std=.8, seed=None, mask=False):
    if seed is None:
        return x + torch.randn(*x.shape, device=x.device) * std
    else:
        noise = torch.zeros(*x.shape, device=x.device)
        noise.normal_(generator=torch.Generator(x.device).manual_seed(seed))
        return x + noise * std

def tensor_sample(*shape, seed=0, distribution="normal", device=device):
    noise = torch.zeros(*shape, device=device)
    if distribution == "normal":
        noise.normal_(generator=torch.Generator(device).manual_seed(seed))
    else:
        raise NotImplementedError()
    return noise


def save_code_under_folder(folder):
    """Saves the code in the current working directory under [folder] if they
    have not already been saved there. For now, all code files are expected
    to be siblings of the current file.
    """
    code_folder = f"{folder}/code"
    _ = conditional_make_folder(code_folder)
    files = sorted([f for f in os.listdir(os.path.dirname(__file__))
        if f.endswith(".py")])
    file_for_diffs = ""
    for f in files:
        with open(f, "r") as opened_file:
            code = opened_file.read()
            file_for_diffs += f"---\n{os.path.basename(f)}\n---\n{code}\n\n\n"
            with open(f"{code_folder}/{f}", "w+") as file_to_write:
                file_to_write.write(code)
    with open(f"{folder}/all_code.txt", "w+") as f:
        f.write(file_for_diffs)

def save_state(model, optimizer, args, epoch, folder, save_latest=False):
    """Saves [model], [optimizer], [args], and [epoch] along with Python, NumPy,
    and PyTorch random seeds to 'folder/epoch.pt'.

    Args:
    model       -- model to be saved
    optimizer   -- optimizer to be saved
    args        -- argparse Namespace used to create run
    epoch       -- epoch number to save with
    folder      -- folder inside which to save everything
    save_latest -- save to 'FOLDER/EPOCH_latest.pt' and delete all other
                    'FOLDER/*_latest.pt' files.
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
    if save_latest:
        for t in [f for f in os.listdir(folder) if f.endswith("_latest.pt")]:
            os.remove(f"{folder}/{t}")
        torch.save(state_dict, f"{folder}/{epoch}_latest.pt")
    else:
        torch.save(state_dict, f"{folder}/{epoch}.pt")
    model.to(device if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Seeds the program to use seed [seed]."""
    if isinstance(seed, int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
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

def compose(*args):
    """Returns a function that is the composition of [args]."""
    def f(x):
        for a in args:
            x = a(x)
        return x
    return f


def hierararchical_hasattr(obj, attrs_list):
    """Returns if the sequence of attributes in [attrs_list] accesses something
    in object [obj]. Example: if the code `x = obj.a.b.c` would work, then
    hierararchical_hasattr(x, ['a', 'b', 'c']) would be True.
    """
    x = obj
    for attr in attrs_list:
        if hasattr(x, attr):
            x = getattr(x, attr)
        else:
            return False
    return True

def images_to_pil_image(images, sigmoid=False, scale_each=True):
    if len(images.shape) == 5:
        pass
    elif len(images.shape) == 4 and images.shape[-1] == 28:
        images = images.view(images.shape[0], images.shape[1], 1, images.shape[-2], images.shape[-1])
    else:
        raise NotImplementedError(f"Wrong shape: {images.shape}")

    nrow = images.shape[1]
    ncol = images.shape[0]
    images = images.view(nrow * ncol, *images.shape[2:])
    
    images = torch.sigmoid(images) if sigmoid else images
    grid = torchvision.utils.make_grid(images,
        nrow=nrow, ncol=ncol,
        scale_each=scale_each,
        normalize=True)
    ndarr = torch.clip((grid * 255), 0, 255).to("cpu", torch.uint8)
    ndarr = ndarr.permute(1, 2, 0).to("cpu").numpy()
    return Image.fromarray(ndarr)

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

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

def split_by_param_names(model, *param_names):
    name2params = {p: [] for p in param_names} | {"default": []}
    for k,v in model.named_parameters():
        found_custom_name = False
        for p in param_names:
            if p in k:
                name2params[p].append(v)
                found_custom_name = True
                break
        if not found_custom_name:
            name2params["default"].append(v)

    return [{"params": p, "name": n} for n,p in name2params.items()]

class StepScheduler:
    """StepLR but with easier control.
    
    Args:
    optimizer   -- optimizer to step
    lrs         -- list where sequential pairs of elements describe a step index
                    and the learning rate for that step and subsequent steps
                    until a new learning rate is specified
    last_epoch  -- the last run step
    named_lr_muls -- dictionary mapping names to multipliers on learning
                            rates specified in lrs. This is a simple and convenient way to have different learning rates for different layers
    """
    def __init__(self, optimizer, lrs, last_epoch=-1, named_lr_muls={}):
        super(StepScheduler, self).__init__()
        self.optimizer = optimizer
        self.named_lr_muls = named_lr_muls
        
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

            if "name" in pg and pg["name"] in self.named_lr_muls:
                pg["lr"] = pg["lr"] * self.named_lr_muls[pg["name"]]

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