from collections import OrderedDict
from functools import partial
import torch
import torch.nn as nn
from tqdm import tqdm as tqdm

import Utils

device = Utils.device

def get_model(args, imle=False, **kwargs):
    """Returns the model architecture specified in argparse Namespace [args].
    [imle] toggles whether the model should be created for IMLE or not.
    """
    if args.arch == "mlp" and imle:
        return IMLE_DAE_MLP(args, **kwargs)
    elif args.arch == "conv" and imle:
        return IMLE_DAE_Conv(args, **kwargs)
    elif args.arch == "1dbasic" and imle:
        return IMLEOneDBasic(args, **kwargs)
    elif args.arch == "mlp" and not imle:
        return DAE_MLP(args)
    elif args.arch == "conv" and not imle:
        return DAE_Conv(args)
    else:
        raise NotImplementedError()

def get_loss_fn(args, reduction="mean"):
    """Returns the loss function given by [args] with reduction [reduction]."""
    if args.loss == "bce":
        return nn.BCEWithLogitsLoss(reduction=reduction)
    elif args.loss == "mse":
        return nn.MSELoss(reduction=reduction)
    else:
        raise NotImplementedError()

def get_fusion(args, ignore_latents=False):
    if args.fusion == "adain":
        return IgnoreLatentAdaIN(args) if ignore_latents else AdaIN(args)
    elif args.fusion == "true_adain":
        return TrueAdaIN(args)

class MLP(nn.Module):
    def __init__(self, in_dim, h_dim=256, out_dim=42, layers=2, lrmul=0.01,
        act_type="leakyrelu", equalized_lr=False, end_with_act=True):
        super(MLP, self).__init__()

        if layers == 0:
            self.model = nn.Identity()
        elif layers == 1 and end_with_act:
            self.model = nn.Sequential(
                get_lin_layer(in_dim, out_dim, equalized_lr=equalized_lr, lrmul=lrmul),
                get_act(act_type))
        elif layers == 1 and not end_with_act:
            self.model = get_lin_layer(in_dim, out_dim,
                equalized_lr=equalized_lr)
        elif layers > 1:
            layer1 = get_lin_layer(in_dim, h_dim, equalized_lr=equalized_lr, lrmul=lrmul)
            mid_layers = [get_lin_layer(h_dim, h_dim, equalized_lr=equalized_lr, lrmul=lrmul)
                for _ in range(layers - 2)]
            layerN = get_lin_layer(h_dim, out_dim, equalized_lr=equalized_lr, lrmul=lrmul)
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

#################################################################################
# MNIST Architectures
#################################################################################
class MLPEncoder(MLP):

    def __init__(self, in_dim=784, encoder_h_dim=1024, feat_dim=64, leaky_relu=False, num_encoder_layers=2, **kwargs):
        super(MLPEncoder, self).__init__(in_dim=in_dim,
            h_dim=encoder_h_dim,
            out_dim=feat_dim,
            layers=num_encoder_layers,
            act_type="leakyrelu" if leaky_relu else "relu")
        self.feat_dim = feat_dim
    
    def forward(self, x): return self.model(torch.flatten(x, start_dim=1))

class MLPDecoder(MLP):

    def __init__(self, feat_dim=64, decoder_h_dim=1024, out_dim=784, num_decoder_layers=2, **kwargs):
        super(MLPDecoder, self).__init__(in_dim=feat_dim,
            h_dim=decoder_h_dim,
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

    def __init__(self, args, ignore_latents=False, in_out_dim=784, **kwargs):
        super(IMLE_DAE_MLP, self).__init__()
        self.args = args
        self.in_out_dim = in_out_dim
        self.ignore_latents = ignore_latents
        self.encoder = MLPEncoder(in_dim=self.in_out_dim, **vars(args))
        self.decoder = MLPDecoder(out_dim=self.in_out_dim, **vars(args))
        self.ada_in = get_fusion(args)
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
            in_out_dim=self.in_out_dim,
            ignore_latents=True)
        ignore_latent_imle_dae_mlp.load_state_dict(self.state_dict(), strict=False)
        return ignore_latent_imle_dae_mlp

#################################################################################
# CIFAR Architectures
#################################################################################
class ConvEncoder(nn.Module):
    def __init__(self, args, **kwargs):
        super(ConvEncoder, self).__init__()
        self.feat_dim = args.feat_dim
        hc = args.encoder_h_dim
        self.model = nn.Sequential(
            nn.Conv2d(3, hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, hc, 3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(hc, 2 * hc, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * hc, 2 * hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * hc, 4 * hc, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * hc, 4 * hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(256 * hc, args.feat_dim), # harcoded for 32x32 inputs
            nn.ReLU(inplace=True))
        self.model.apply(partial(init_weights_fn, args))

    def forward(self, x): return self.model(x)

class ConvDecoder(nn.Module):
    def __init__(self, args, **kwargs):
        super(ConvDecoder, self).__init__()
        self.feat_dim = args.feat_dim
        hc = args.decoder_h_dim
        self.model = nn.Sequential(
            nn.Linear(args.feat_dim, 4 * hc*8*8),
            nn.ReLU(inplace=True),
            nn.Unflatten(-1, (4 * hc, 8, 8)),
            nn.ConvTranspose2d(4 * hc, 4 * hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(4 * hc, 2 * hc, 3, padding=1, stride=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * hc, 2 * hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(2 * hc, hc, 3, padding=1, stride=2, output_padding=1),
            nn.ReLU(inplace=True), 
            nn.ConvTranspose2d(hc, hc, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hc, 3, 3, padding=1))
        self.model.apply(partial(init_weights_fn, args))

    def forward(self, x): return self.model(x)

class DAE_Conv(nn.Module):
    """Adapted from https://blog.paperspace.com/convolutional-autoencoder/."""
    def __init__(self, args, **kwargs):
        super(DAE_Conv, self).__init__()
        self.args = args
        self.encoder = ConvEncoder(args)
        self.decoder = ConvDecoder(args)
        self.feat_dim = args.feat_dim

    def forward(self, x): return self.decoder(self.encoder(x))

class IMLE_DAE_Conv(nn.Module):
    """Adapted from https://blog.paperspace.com/convolutional-autoencoder/."""
    def __init__(self, args, **kwargs):
        super(IMLE_DAE_Conv, self).__init__()
        self.args = args
        self.encoder = ConvEncoder(args)
        self.decoder = ConvDecoder(args)
        self.ada_in = get_fusion(args)
        self.feat_dim = args.feat_dim
        self.latent_dim = args.latent_dim

    def forward(self, x, z=None, num_z=1, seed=None):
        if z is None:
            z = self.get_codes(len(x) * num_z, device=x.device, seed=seed)
        fx = self.decoder(self.ada_in(self.encoder(x), z))
        return fx.view(len(z), * x.shape[1:])

    def get_codes(self, bs, device="cpu", seed=None):
        if seed is None:
            return torch.randn(bs, self.latent_dim, device=device)
        else:
            z = torch.zeros(bs, self.latent_dim, device=device)
            z.normal_(generator=torch.Generator(device).manual_seed(seed))
            return z
    
    def to_encoder_with_ada_in(self, use_mean_representation=False):
        return EncoderWithAdaIn(self.encoder, self.ada_in,
            use_mean_representation=use_mean_representation)

def init_weights_fn(args, module):
    if args.init == "baseline" and args.arch == "conv":
        if isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

#################################################################################
# Misc/General
#################################################################################
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
        return get_codes(bs, self.latent_dim, device=device, seed=seed)

    def forward(self, x, z=None, num_z=1, seed=None):
        in_shape = x.shape[1:]
        
        if self.use_mean_representation:
            if z is None:
                num_z = 64
                z = self.get_codes(len(x) * num_z,  device=x.device, seed=seed)

            bs = x.shape[0]
            fx = self.ada_in(self.encoder(x), z)
            fx = fx.view(bs, num_z, -1)
            fx = fx.mean(dim=1)
            return fx
        else:
            if z is None:
                z = self.get_codes(len(x) * num_z, device=x.device, seed=seed)
            fx = self.encoder(x)
            return self.ada_in(fx, z)

class AdaIN(nn.Module):

    def __init__(self, args):
        super(AdaIN, self).__init__()
        self.args = args
        self.feat_dim = args.feat_dim
        self.normalize_z = args.normalize_z
        self.mapping_net_h_dim = args.mapping_net_h_dim
        self.mapping_net_layers = args.mapping_net_layers
        self.latent_dim = args.latent_dim
        self.adain_x_norm = args.adain_x_norm

        layers = []
        if args.normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=0)))
        layers.append(("mapping_net", MLP(in_dim=args.latent_dim,
            h_dim=args.mapping_net_h_dim,
            layers=args.mapping_net_layers,
            out_dim=args.feat_dim * 2,
            act_type=args.mapping_net_act,
            equalized_lr=args.mapping_net_eqlr,
            lrmul=args.mapping_net_lrmul,
            end_with_act=False)))

        self.model = nn.Sequential(OrderedDict(layers))

        if args.adain_x_mod == "linear":
            self.x_modification_layer = nn.Linear(self.feat_dim, self.feat_dim)
        else:
            self.x_modification_layer = nn.Identity()
            
        if args.adain_x_norm == "none":
            self.x_norm_layer = nn.Identity()
        elif args.adain_x_norm == "norm":
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
            z = self.model(get_codes(num_z, self.latent_dim, device=device))
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
        ada_in = IgnoreLatentAdaIN(self.args)
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

class TrueAdaIN(nn.Module):

    def __init__(self, args):
        super(TrueAdaIN, self).__init__()
        self.args = args
        self.feat_dim = args.feat_dim
        self.normalize_z = args.normalize_z
        self.mapping_net_h_dim = args.mapping_net_h_dim
        self.mapping_net_layers = args.mapping_net_layers
        self.latent_dim = args.latent_dim

        layers = []
        if args.normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=0)))
        layers.append(("mapping_net", MLP(in_dim=args.latent_dim,
            h_dim=args.mapping_net_h_dim,
            layers=args.mapping_net_layers,
            out_dim=args.feat_dim * 2,
            act_type=args.mapping_net_act,
            equalized_lr=args.mapping_net_eqlr,
            lrmul=args.mapping_net_lrmul,
            end_with_act=False)))

        self.model = nn.Sequential(OrderedDict(layers))

    def get_z_stats(self, num_z=2048, device="cpu"):
        """Returns the mean shift and scale used in the AdaIN, with the mean
        taken over [num_z] different latent codes.
        """
        with torch.no_grad():
            z = self.model(get_codes(num_z, self.latent_dim, device=device))
            z_shift, z_scale = z[:, :self.feat_dim], z[:, self.feat_dim:]
            return torch.mean(z_shift, dim=0), torch.std(z_shift, dim=0), torch.mean(z_scale, dim=0), torch.std(z_scale, dim=0)

    def forward(self, x, z):
        """
        Args:
        x   -- NxD image features
        z   -- (N*k)xCODE_DIM latent codes
        """
        z = self.model(z)
        z_shift = torch.mean(z[:, :self.feat_dim], dim=1, keepdim=True)
        z_scale = torch.std(z[:, self.feat_dim:], dim=1, keepdim=True)
        x = torch.repeat_interleave(x, z.shape[0] // x.shape[0], dim=0)
        x_normalized = (x - torch.mean(x, dim=1, keepdim=True)) / torch.std(x, dim=1, keepdim=True)
        return z_scale * x_normalized + z_shift

class OneDFakeAdaIN(nn.Module):
    """A mapping net that can accept and ignore a value [x]."""
    
    def __init__(self, args):
        super(OneDFakeAdaIN, self).__init__()
        layers = []
        if args.normalize_z:
            layers.append(("normalize_z", PixelNormLayer(epsilon=0)))
        layers.append(("mapping_net", MLP(in_dim=args.latent_dim,
            h_dim=args.mapping_net_h_dim,
            layers=args.mapping_net_layers,
            out_dim=1,
            act_type=args.mapping_net_act,
            equalized_lr=args.mapping_net_eqlr,
            lrmul=args.mapping_net_lrmul,
            end_with_act=False)))
        self.model = nn.Sequential(OrderedDict(layers))

    def forward(self, *args): return self.model(args[-1])

class IMLEOneDBasic(nn.Module):

    def __init__(self, args, in_out_dim=1):
        super(IMLEOneDBasic, self).__init__()
        self.latent_dim = args.latent_dim
        self.a = nn.Parameter(torch.Tensor([[0.]]))
        self.b = nn.Parameter(torch.Tensor([[0.]]))
        self.ada_in = OneDFakeAdaIN(args)

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
        
        fx = self.a * x + self.b
        fx = torch.repeat_interleave(fx, z.shape[0] // x.shape[0], dim=0)
        return fx + self.ada_in(z)


    