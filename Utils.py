from collections import OrderedDict, defaultdict
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import io

from tqdm import tqdm

def conditional_make_folder(f):
    try:
        os.makedirs(f)
    except:
        pass

def with_noise(x, std=.8, seed=None):
    if seed is None:
        return x + torch.randn(*x.shape, device=x.device) * std
    else:
        noise = torch.zeros(*x.shape, device=x.device)
        noise.normal_(generator=torch.Generator(x.device).manual_seed(seed))
        return x + noise * std

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

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
    """Returns tensor datastructure [images] as a PIL image."""
    if images.shape[-1] == 784 and len(images.shape) == 3:
        images = images.view(images.shape[0], images.shape[1], 28, 28)
    elif images.shape[-1] == 784 and len(images.shape) == 2:
        images = images.view(images.shape[0], 1, 28, 28)
    elif images.shape[-1] == 28 and len(images.shape) == 3:
        images = images.view(images.shape[0], 1, 28, 28)
    elif images.shape[-1] == 28 and len(images.shape) == 4:
        pass
    elif isinstance(images, list):
        pass
    else:
        raise NotImplementedError()


    fig, axs = plt.subplots(ncols=max([len(image_row) for image_row in images]),
        nrows=len(images),
        squeeze=False)

    for i,images_row in enumerate(images):
        for j,image in enumerate(images_row):
            image = torch.clip((image * 255), 0, 255).int().cpu()
            axs[i, j].imshow(np.asarray(image), cmap='Greys_r')
            axs[i, j].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    buf = io.BytesIO()
    fig.savefig(buf, dpi=256)
    buf.seek(0)
    plt.close("all")
    return Image.open(buf)

def de_dataparallel(model):
    if isinstance(model, nn.DataParallel):
        return model.module
    else:
        return model

class StepScheduler:
    """StepLR but with easier control.
    
    Args:
    optimizer   -- optimizer to step
    lrs         -- list where sequential pairs of elements describe a step index
                    and the learning rate for that step and subsequent steps
                    until a new learning rate is specified
    last_epoch  -- the last run step
    """
    def __init__(self, optimizer, lrs, args=None, last_epoch=-1):
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