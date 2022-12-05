import numpy as np
import random
import torch
from torchvision import transforms

def with_noise(x, std=.4):
    return x + torch.randn(*x.shape, device=x.device) * std

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor

transforms_tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: min_max_normalization(x, 0, 1)),
    transforms.Lambda(lambda x: torch.round(x))
])

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)