import numpy as np
import os
from PIL import Image
import random
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import io

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

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

transforms_mnist = transforms.Compose([
    transforms.Lambda(lambda x: min_max_normalization(x, 0, 1)),
    transforms.Lambda(lambda x: torch.round(x))
])

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