import argparse
import os

def dataset_spec_type(s):
    datasets = ["mnist", "cifar10"]
    if os.path.exists(s) or s.startswith(f"$SLURM_TMPDIR") or s in datasets:
        return s
    else:
        raise FileNotFoundError(s)

def int_or_all(s):
    if s.isdigit():
        return int(s)
    elif s == "all":
        return s
    else:
        raise argparse.ArgumentTypeError(f"Must be int or 'all'")

def parser_with_default_args(P):
    P.add_argument("--wandb", default="disabled", choices=["disabled", "online", "offline"],
        help="WandB mode")
    P.add_argument("--suffix", default=None,
        help="Optional suffix")
    P.add_argument("--seed", default=0, type=int,
        help="Seed")
    P.add_argument("--save_folder", default=f"{os.path.abspath(os.path.dirname(__file__))}",
        help="Absolute path to where to save")
    P.add_argument("--num_workers", default=20, type=int,
        help="Number of DataLoader workers")
    P.add_argument("--gpus", type=int, nargs="+", default=[0, 1],
        help="List of GPU IDs")
    P.add_argument("--resume", type=str, default=None,
        help="Path to file to resume from")
    return P

def parser_with_logging_args(P):
    P.add_argument("--script", default=None,
        help="Name of the script being run. Individual scripts should set this if not None.")
    P.add_argument("--job_id", type=str, default=None,
        help="SLURM job ID")
    P.add_argument("--uid", default=None,
        help="WandB UID. Specify only to resume an existing run.")
    P.add_argument("--eval_iter", default=1, type=int,
        help="Evaluate on the proxy task every EVAL_ITER epochs/samplings.")
    P.add_argument("--save_iter", default=0, type=int,
        help="Save the run every SAVE_ITER epochs/samplings.")
    P.add_argument("--save_epochs", nargs="*", default=[], type=int,
        help="List of epoch indices on which to necessarily save the model.")
    P.add_argument("--probe_iter", default=5, type=int,
        help="Probe every PROBE_ITER epochs/samplings. Must be multiple of EVAL_ITER")
    P.add_argument("--probe_eval_iter", default=10, type=int,
        help="Evaluate the probe every PROBE_EVAL_ITER epochs during probing")
    P.add_argument("--num_eval_images", type=int, default=10,
        help="Number of images generate to generate from when logging images during evaluation")
    P.add_argument("--num_eval_samples", type=int, default=10,
        help="Number of latent codes per imagewhen logging images during evaluation")
    return P

def parser_with_data_args(P):
    P.add_argument("--data_tr", required=True, type=dataset_spec_type,
        help="Training data. 'mnist', 'cifar10', or path to a file.")
    P.add_argument("--data_val", default=None, required=False, type=dataset_spec_type,
        help="Validation data. If not specified, uses the data excluded through N_WAY and N_SHOT settings.")
    P.add_argument("--n_shot", default="all", type=int_or_all,
        help="Number of examples per class. -1 or 'all' for all")
    P.add_argument("--n_way", default="all", type=int_or_all,
        help="Number of classes. -1 or 'all' for all")
    return P

def parser_with_training_args(P):
    P.add_argument("--arch", choices=["mlp", "mlp_bn", "mlp_ln"], default="mlp",
        help="Model architecture")
    P.add_argument("--feat_dim", type=int, default=64,
        help="Dimensionality of the features extracted by the model")
    P.add_argument("--leaky_relu", type=int, default=0, choices=[0, 1],
        help="Use LeakyReLU instead of ReLU")
    P.add_argument("--lrs", default=[0, 1e-5], type=float, nargs="*",
        help="Learning rates. Even indices give step indices, odd indices give the learning rate to start at the step given at the prior index.")
    P.add_argument("--epochs",type=int, default=5000,
        help="Number of epochs/samplings")
    P.add_argument("--bs",type=int, default=1000,
        help="Batch size")
    P.add_argument("--std",type=float, default=.8,
        help="Noise standard deviation")
    return P

def parser_with_probe_args(P):
    P.add_argument("--probe_lrs", default=[0, 1e-3], type=float, nargs="*",
        help="Learning rates. Even indices give step indices, odd indices give the learning rate to start at the step given at the prior index.")
    P.add_argument("--probe_epochs",type=int, default=100,
        help="Number of epochs for the probe")
    P.add_argument("--probe_linear", choices=[0, 1], default=0, type=int,
        help="Whether to include a linear probe")
    P.add_argument("--probe_mlp", choices=[0, 1], default=0, type=int,
        help="Whether the probe should include an MLP")
    P.add_argument("--probe_include_codes", choices=[0, 1, 2], default=0, type=int,
        help="Whether the probe should include noise. 2 does it both ways")
    P.add_argument("--probe_verbosity", choices=[0, 1], default=1, type=int,
        help="Probe verbosity.")
    P.add_argument("--probe_normalize_feats", choices=[0, 1], default=0, type=int,
        help="Probe verbosity.")
    return P

def parser_with_imle_args(P):
    P.add_argument("--ns", default=128, type=int,
        help="Number of latent codes to take the min over per image")
    P.add_argument("--code_bs",type=int, default=60000,
        help="Number of images to sample codes for at once")
    P.add_argument("--ipe", default=50, type=int,
        help="Number of gradient steps per code.")
    P.add_argument("--latent_arch", choices=["v0", "v1", "v2"], default="v0",
        help="Architecture for noise injection")
    return P
