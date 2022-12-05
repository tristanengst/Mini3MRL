import argparse
import os

def parser_with_default_args(P):
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online"],
        help="WandB mode")
    P.add_argument("--suffix", default=None,
        help="Optional suffix")
    P.add_argument("--eval_iter", default=5, type=int,
        help="Iterations (epochs) between evaluations")
    P.add_argument("--num_workers", default=20, type=int,
        help="Number of DataLoader workers")
    P.add_argument("--seed", default=42, type=int,
        help="Number of DataLoader workers")
    P.add_argument("--save_folder", default=f"{os.path.abspath(os.path.dirname(__file__))}",
        help="Absolute path to where to save")
    return P

def parser_with_training_args(P):
    P.add_argument("--lr", default=1e-3, type=float,
        help="Learning rate")
    P.add_argument("--epochs",type=int, default=100,
        help="Number of epochs")
    P.add_argument("--bs",type=int, default=1000,
        help="Batch size")
    P.add_argument("--subsample",type=int, default=None,
        help="Number of training examples to use")
    P.add_argument("--std",type=float, default=.8,
        help="Noise standard deviation")
    return P

def parser_with_probe_args(P):
    P.add_argument("--probe_lr", default=1e-3, type=float,
        help="Learning rate")
    P.add_argument("--probe_epochs",type=int, default=10,
        help="Number of epochs")
    return P

def parser_with_imle_args(P):
    P.add_argument("--ns", default=32, type=int,
        help="Number of latent codes to take the min over per image")
    P.add_argument("--code_bs",type=int, default=10000,
        help="Number of images to sample codes for at once")
    P.add_argument("--ipe", default=50, type=int,
        help="Number of gradient steps per code.")
    return P
