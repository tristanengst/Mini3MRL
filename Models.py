from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm

class MLPEncoder(nn.Module):

    def __init__(self, in_dim=784, h_dim=1024, out_dim=64):
        super(MLPEncoder, self).__init__()
        self.lin1 = nn.Linear(in_dim, h_dim)
        self.relu = nn.ReLU(True)
        self.lin2 = nn.Linear(h_dim, out_dim)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        fx = self.lin1(x)
        fx = self.relu(fx)
        fx = self.lin2(fx)
        return fx

class DAE(nn.Module):

    def __init__(self):
        super(DAE, self).__init__()
        self.encoder = MLPEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid())
    
    def forward(self, x):
        fx = self.encoder(x)
        fx = self.decoder(fx)
        return fx

class IMLE_DAE(nn.Module):

    def __init__(self):
        super(IMLE_DAE, self).__init__()
        self.code_dim = 512
        self.encoder = MLPEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),
            nn.Sigmoid())
        self.ada_in = AdaIN(64)

        self.code_dim = 512

    def get_codes(self, bs, device="cpu", seed=None):
        """Returns [bs] latent codes to be passed into the model.

        Args:
        bs      -- number of latent codes to return
        device  -- device to return latent codes on
        seed    -- None for no seed (outputs will be different on different
                    calls), or a number for a fixed seed
        """
        if seed is None:
            return torch.randn(bs, self.code_dim, device=device)
        else:
            z = torch.zeros(bs, self.code_dim, device=device)
            z.normal_(generator=torch.Generator(device).manual_seed(seed))
            return z
    
    def forward(self, x, z=None, num_z=1, seed=None):
        if z is None:
            z = self.get_codes(len(x) * num_z,
                device=x.device,
                seed=seed)

        fx = self.encoder(x)
        fx = self.ada_in(fx, z)
        fx = self.decoder(fx)
        return fx

class AdaIN(nn.Module):
    """AdaIN adapted for a transformer. Expects a BSxNPxC batch of images, where
    each image is represented as a set of P tokens, and BSxPxZ noise. This noise
    is mapped to be BSx1x2C. These are used to scale the image patches, ie. in
    the ith image, the kth element of the jth patch is scaled identically to the
    kth element of any other patch in that image.
    """
    def __init__(self, c, epsilon=1e-8, act_type="leakyrelu", normalize_z=True):
        super(AdaIN, self).__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.c = c

        layers = []
        if normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=epsilon)))
        layers.append(("mapping_net", MLP(in_dim=512,
            h_dim=512,
            layers=8,
            out_dim=self.c * 2,
            equalized_lr=True,
            act_type=act_type)))

        self.model = nn.Sequential(OrderedDict(layers))

        
    def get_latent_spec(self, x): return (512,)

    def forward(self, x, z):
        """
        Args:
        x   -- image features
        z   -- latent codes
        """
        z = self.model(z)
        z_mean = z[:, :self.c]
        z_std = z[:, self.c:]

        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        result = z_mean + x * (1 + z_std)
        return result

def get_act(act_type):
    """Returns an activation function of type [act_type]."""
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=.2)
    else:
        raise NotImplementedError(f"Unknown activation '{act_type}'")

def get_lin_layer(in_dim, out_dim, equalized_lr=True, bias=True, **kwargs):
    """
    """
    if equalized_lr:
        return EqualizedLinear(in_dim, out_dim, bias=bias, **kwargs)
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)


class PixelNormLayer(nn.Module):
    """From https://github.com/huangzh13/StyleGAN.pytorch/blob/b1dfc473eab7c1c590b39dfa7306802a0363c198/models/CustomLayers.py.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        return x * torch.rsqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

class EqualizedLinear(nn.Module):
    """Linear layer with equalized learning rate and custom learning rate multiplier.
    
    From https://github.com/huangzh13/StyleGAN.pytorch/blob/master/models/CustomLayers.py.
    """

    def __init__(self, input_size, output_size, gain=2 ** .5, use_wscale=True, lrmul=.01, bias=True):
        super().__init__()
        he_std = gain * input_size ** (-0.5)
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_mul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_mul = lrmul
        self.weight = torch.nn.Parameter(torch.randn(output_size, input_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_size))
            self.b_mul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        bias = self.bias * self.b_mul if self.bias is not None else self.bias
        return nn.functional.linear(x, self.weight * self.w_mul, bias)

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=256, out_dim=42, layers=4, 
        act_type="leakyrelu", equalized_lr=True, end_with_act=True):
        super(MLP, self).__init__()

        if layers == 1 and end_with_act:
            self.model = nn.Sequential(
                get_lin_layer(in_dim, out_dim, equalized_lr=equalized_lr),
                get_act(act_type))
        elif layers == 1 and not end_with_act:
            self.model = get_lin_layer(in_dim, out_dim,
                equalized_lr=equalized_lr)
        elif layers > 1:
            layer1 = get_lin_layer(in_dim, h_dim, equalized_lr=equalized_lr)
            mid_layers = [get_lin_layer(h_dim, h_dim, equalized_lr=equalized_lr)
                for _ in range(layers - 2)]
            layerN = get_lin_layer(h_dim, out_dim, equalized_lr=equalized_lr)
            linear_layers = [layer1] + mid_layers + [layerN]

            layers = []
            for idx,l in enumerate(linear_layers):
                layers.append(l)
                if end_with_act:
                    layers.append(get_act(act_type))
                elif not end_with_act and idx < len(linear_layers) - 1:
                    layers.append(get_act(act_type))
                else:
                    continue
            
            self.model = nn.Sequential(*layers)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.h_dim = h_dim
                
    def forward(self, x): return self.model(x)