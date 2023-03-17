from collections import OrderedDict
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm

import Utils

device = Utils.device

def get_model(args, imle=False):
    """Returns the model architecture specified in argparse Namespace [args].
    [imle] toggles whether the model should be created for IMLE or not.
    """
    if args.arch == "mlp" and imle:
        return IMLE_DAE_MLP(args)
    elif args.arch == "mlp" and not imle:
        return DAE_MLP(args)
    else:
        raise NotImplementedError()

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=256, out_dim=42, layers=2, 
        act_type="leakyrelu", equalized_lr=False, end_with_act=True):
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

class MLPEncoder(MLP):

    def __init__(self, in_dim=784, h_dim=1024, feat_dim=64, leaky_relu=False, num_encoder_layers=2, **kwargs):
        super(MLPEncoder, self).__init__(in_dim=in_dim,
            h_dim=h_dim,
            out_dim=feat_dim,
            layers=num_encoder_layers,
            act_type="leakyrelu" if leaky_relu else "relu")
        self.feat_dim = feat_dim
    
    def forward(self, x): return self.model(torch.flatten(x, start_dim=1))

class MLPDecoder(MLP):

    def __init__(self, feat_dim=64, h_dim=1024, out_dim=784, num_decoder_layers=2, **kwargs):
        super(MLPDecoder, self).__init__(in_dim=feat_dim,
            h_dim=h_dim,
            out_dim=out_dim,
            layers=num_decoder_layers,
            act_type="relu",
            end_with_act=False)

    def forward(self, x): return self.model(x)

class DAE_MLP(nn.Module):

    def __init__(self, args):
        super(DAE_MLP, self).__init__()
        self.encoder = MLPEncoder(**vars(args))
        self.decoder = MLPDecoder(**vars(args))
    
    def forward(self, x): return self.decoder(self.encoder(x)).view(*x.shape)


class IMLE_DAE_MLP(nn.Module):

    def __init__(self, args, ignore_latents=False, **kwargs):
        super(IMLE_DAE_MLP, self).__init__()
        self.args = args
        self.ignore_latents = ignore_latents
        self.encoder = MLPEncoder(**vars(args))
        self.decoder = MLPDecoder(**vars(args))
        self.ada_in = IgnoreLatentAdaIN(**vars(args)) if ignore_latents else AdaIN(**vars(args))
        self.feat_dim = args.feat_dim
        self.latent_dim = args.latent_dim

    def get_codes(self, bs, device="cpu", seed=None):
        """Returns [bs] latent codes to be passed into the model.

        Args:
        bs      -- number of latent codes to return
        device  -- device to return latent codes on
        seed    -- None for no seed (outputs will be different on different
                    calls), or a number for a fixed seed
        """
        if seed is None:
            return torch.randn(bs, self.latent_dim, device=device)
        else:
            z = torch.zeros(bs, self.latent_dim, device=device)
            z.normal_(generator=torch.Generator(device).manual_seed(seed))
            return z
    
    def forward(self, x, z=None, num_z=1, seed=None):
        if z is None:
            z = self.get_codes(len(x) * num_z, device=x.device, seed=seed)

        fx = self.decoder(self.ada_in(self.encoder(x), z))
        return fx.view(len(z), * x.shape[1:])

    def to_encoder_with_ada_in(self, use_mean_representation=False):
        return EncoderWithAdaIn(self.encoder, self.ada_in,
            use_mean_representation=use_mean_representation)

    def to_ignore_latent_imle_dae_mlp(self):
        ignore_latent_imle_dae_mlp = self.__class__(self.args,
            ignore_latents=True)
        ignore_latent_imle_dae_mlp.load_state_dict(self.state_dict(), strict=False)
        return ignore_latent_imle_dae_mlp

def get_codes(bs, code_dim, device="cpu", seed=None):
    """Returns [bs] latent codes to be passed into the model.

    Args:
    bs          -- number of latent codes to return
    code_dim    -- dimension of input latent codes
    device      -- device to return latent codes on
    seed        -- None for no seed (outputs will be different on different
                    calls), or a number for a fixed seed
    """
    if seed is None:
        return torch.randn(bs, code_dim, device=device)
    else:
        z = torch.zeros(bs, code_dim, device=device)
        z.normal_(generator=torch.Generator(device).manual_seed(seed))
        return z

class EncoderWithAdaIn(nn.Module):

    def __init__(self, encoder, ada_in, use_mean_representation=False):
        super(EncoderWithAdaIn, self).__init__()
        self.encoder = encoder
        self.ada_in = ada_in
        self.latent_dim = self.ada_in.feat_dim
        self.feat_dim = self.encoder.feat_dim

        self.use_mean_representation = use_mean_representation

    def get_codes(self, bs, device="cpu", seed=None):
        """Returns [bs] latent codes to be passed into the model.

        Args:
        bs      -- number of latent codes to return
        device  -- device to return latent codes on
        seed    -- None for no seed (outputs will be different on different
                    calls), or a number for a fixed seed
        """
        if seed is None:
            return torch.randn(bs, self.latent_dim, device=device)
        else:
            z = torch.zeros(bs, self.latent_dim, device=device)
            z.normal_(generator=torch.Generator(device).manual_seed(seed))
            return z

    def forward(self, x, z=None, num_z=1, seed=None):
        in_shape = x.shape[1:]
        
        if self.use_mean_representation:
            if z is None:
                num_z = 64
                z = self.get_codes(len(x) * num_z,  device=x.device, seed=seed)

            bs = x.shape[0]
            fx = self.encoder(x)
            fx = self.ada_in(fx, z)
            fx = fx.view(bs, num_z, -1)
            fx = fx.mean(dim=1)
            return fx
        else:
            if z is None:
                z = self.get_codes(len(x) * num_z, device=x.device, seed=seed)
            fx = self.encoder(x)
            return self.ada_in(fx, z)

class AdaIN(nn.Module):

    def __init__(self, feat_dim=64, adain_x_norm=None, normalize_z=True, mapping_net_layers=8, mapping_net_h_dim=512, latent_dim=512, **kwargs):
        super(AdaIN, self).__init__()
        self.feat_dim = feat_dim
        self.normalize_z = normalize_z
        self.mapping_net_h_dim = mapping_net_h_dim
        self.mapping_net_layers = mapping_net_layers
        self.latent_dim = latent_dim
        self.adain_x_norm = adain_x_norm

        layers = []
        if normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=0)))
        layers.append(("mapping_net", MLP(in_dim=latent_dim,
            h_dim=mapping_net_h_dim,
            layers=mapping_net_layers,
            out_dim=self.feat_dim * 2,
            equalized_lr=True,
            end_with_act=False,
            act_type="leakyrelu")))

        self.model = nn.Sequential(OrderedDict(layers))

        self.x_modification_layer = nn.Linear(self.feat_dim, self.feat_dim)

        if adain_x_norm == "none":
            self.x_norm_layer = nn.Identity()
        elif adain_x_norm == "norm":
            self.x_norm_layer = NormLayer()
        else:
            raise NotImplementedError()

        self.register_buffer("z_scale_mean", torch.zeros(self.feat_dim))
        self.register_buffer("z_shift_mean", torch.zeros(self.feat_dim))
        self.init_constants()

    def get_z_stats(self, num_z=2048, device="cpu"):
        """Returns the mean shift and scale used in the AdaIN, with the mean
        taken over [num_z] different latent codes.
        """
        with torch.no_grad():
            z = self.model(get_codes(num_z, 512, device=device))
            z_shift, z_scale = z[:, :self.feat_dim], z[:, self.feat_dim:]
            return torch.mean(z_shift, dim=0), torch.std(z_shift, dim=0), torch.mean(z_scale, dim=0), torch.std(z_scale, dim=0)

    def init_constants(self, num_z=2048):
        """Sets the [z_shift_mean] and [z_shift_scale] constants."""
        self.z_shift_mean, _, self.z_scale_mean, _ = self.get_z_stats(num_z=num_z)

    def forward(self, x, z):
        """
        Args:
        x   -- NxD image features
        z   -- (N*k)xCODE_DIM latent codes
        """
        z = self.model(z)
        z_shift = z[:, :self.feat_dim] - self.z_shift_mean
        z_scale = z[:, self.feat_dim:] - self.z_scale_mean

        z_scale = torch.nn.functional.relu(z_scale)
        x = self.x_modification_layer(x)
        x = self.x_norm_layer(x)

        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        result = z_shift + x * (1 + z_scale)
        return result

    def to_ignore_latent_ada_in(self):
        ada_in = IgnoreLatentAdaIN(feat_dim=self.feat_dim,
            normalize_z=self.normalize_z,
            mapping_net_h_dim=self.mapping_net_h_dim,
            mapping_net_layers=self.mapping_net_layers,
            latent_dim=self.latent_dim,
            adain_x_norm=self.adain_x_norm)
        ada_in.load_state_dict(self.state_dict(), strict=False)
        return ada_in

class IgnoreLatentAdaIN(AdaIN):
    def __init__(self, *args, **kwargs): super(IgnoreLatentAdaIN, self).__init__(*args, **kwargs)

    def forward(self, x, z):
        x = self.x_modification_layer(x)
        x = self.adain_x_norm(x)
        return torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)

def get_act(act_type):
    """Returns an activation function of type [act_type]."""
    if act_type == "gelu":
        return nn.GELU()
    elif act_type == "leakyrelu":
        return nn.LeakyReLU(negative_slope=.2)
    elif act_type == "relu":
        return nn.ReLU(True)
    else:
        raise NotImplementedError(f"Unknown activation '{act_type}'")

def get_lin_layer(in_dim, out_dim, equalized_lr=True, bias=True, **kwargs):
    """
    """
    if equalized_lr:
        return EqualizedLinear(in_dim, out_dim, bias=bias, **kwargs)
    else:
        return nn.Linear(in_dim, out_dim, bias=bias)

class NormLayer(nn.Module):
    """Neural network layer that normalizes its input to have unit norm.
    Normalization occurs over the last dimension of the sample.
    """
    def __init__(self): super(NormLayer, self).__init__()
    def forward(self, x): return nn.functional.normalize(x, dim=-1).view(*x.shape)


class PixelNormLayer(nn.Module):
    """From https://github.com/huangzh13/StyleGAN.pytorch/blob/b1dfc473eab7c1c590b39dfa7306802a0363c198/models/CustomLayers.py.
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x):
        assert len(x.shape) == 2, f"{x.shape}"
        return x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.epsilon)

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
