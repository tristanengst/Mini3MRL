import argparse
import os

def data_type(s):
    datasets = ["mnist", "cifar10"]
    if os.path.exists(s) or s.startswith(f"$SLURM_TMPDIR") or s in datasets:
        return s
    else:
        raise FileNotFoundError(s)

def parser_with_default_args(P):
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"],
        help="WandB mode")
    P.add_argument("--suffix", default=None,
        help="Optional suffix")
    P.add_argument("--seed", default=0, type=int,
        help="Seed")
    P.add_argument("--save_folder", default=f"{os.path.abspath(os.path.dirname(__file__))}",
        help="Absolute path to where to save")
    P.add_argument("--uid", default=None,
        help="WandB UID")
    P.add_argument("--script", default=None,
        help="Name of the script being run")
    P.add_argument("--job_id", type=str, default=None,
        help="SLURM job ID")
    P.add_argument("--num_workers", default=20, type=int,
        help="Number of DataLoader workers")
    P.add_argument("--gpus", type=int, nargs="+", default=[0],
        help="List of GPU IDs")
    P.add_argument("--resume", type=str, default=None,
        help="Path to file to resume from")
    return P

def parser_with_logging_args(P):
    P.add_argument("--eval_iter", default=1, type=int,
        help="Evaluate every EVAL_ITER epochs/samplings.")
    P.add_argument("--save_iter", default=-1, type=int,
        help="Save every SAVE_ITER epochs/samplings.")
    return P

def parser_with_data_args(P):
    P.add_argument("--data_tr", required=True,
        help="Training data")
    P.add_argument("--data_val", default=None, required=False,
        help="Validation data. If not specified")
    P.add_argument("--n_shot", default=-1, type=int,
        help="Number of examples per class. -1 for all")
    P.add_argument("--n_way", default=-1, type=int,
        help="Number of classes. -1 for all")
    return P

def parser_with_training_args(P):
    P.add_argument("--lr", default=1e-3, type=float,
        help="Learning rate")
    P.add_argument("--scheduler", default="constant", choices=["step", "constant"],
        help="Learning rate")
    P.add_argument("--epochs",type=int, default=1000,
        help="Number of epochs")
    P.add_argument("--bs",type=int, default=1000,
        help="Batch size")
    P.add_argument("--std",type=float, default=.8,
        help="Noise standard deviation")
    return P

def parser_with_probe_args(P):
    P.add_argument("--probe_lr", default=1e-3, type=float,
        help="Learning rate")
    P.add_argument("--probe_epochs",type=int, default=25,
        help="Number of epochs")
    return P

def parser_with_imle_args(P):
    P.add_argument("--ns", default=128, type=int,
        help="Number of latent codes to take the min over per image")
    P.add_argument("--code_bs",type=int, default=60000,
        help="Number of images to sample codes for at once")
    P.add_argument("--ipe", default=50, type=int,
        help="Number of gradient steps per code.")
    return P
