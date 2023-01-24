import argparse
import os
from tqdm import tqdm

import DAE
import IMLE_DAE

chunked_scripts = []

def unparse_args(args):
    """Returns [args] as a string that can be parsed again."""
    s = ""
    for k,v in vars(args).items():
        if isinstance(v, (list, tuple)):
            s += f" --{k} {' '.join([str(v_) for v_ in v])}"
        elif v is None:
            continue
        else:
            s += f" --{k} {v}"
    return s


def get_args_with_data_on_node(args, arg_names_to_move, out_dir="$SLURM_TMPDIR"):
    """Returns an (args, cmd) tuple where [args] is [args] modified to have the
    value in [args] of each element of [arg_names_to_move] listed inside
    [out_dir], and [cmd] is a string giving commands to move the files there.
    """
    s = ""
    args = vars(args)
    for a in arg_names_to_move:
        if (a in args and isinstance(args[a], str)
            and os.path.exists(args[a])
            and not "mnist" in args[a] and not "cifar" in args[a]):
            s += f"rsync -rav --relative {args[a]} {out_dir}/\n"
            args[a] = f"{out_dir}/{args[a]}".replace("//", "/").strip("/")
        else:
            continue

    return argparse.Namespace(**args), f"{s}\n"

def get_slurm_args():
    P = argparse.ArgumentParser()
    P.add_argument("script",
        help="Script to run")
    P.add_argument("--time", default="1:00:00", type=str,
        help="String giving time for the SLURM job")
    P.add_argument("--account", default="rrg-keli", choices=["def-keli", "rrg-keli"],
        help="String giving time for the SLURM job")
    P.add_argument("--parallel", default=1, type=int,
        help="How many jobs to run in parallel")
    P.add_argument("--first_chunk", default=0, type=int,
        help="Zero-indexed index of first chunk to run")
    P.add_argument("--last_chunk", default=0, type=int,
        help="Zero-indexed index of last chunk to run, (ie. 9 to run 10 chunks")
    P.add_argument("--env", default="conda", choices=["conda", "pip"],
        help="Python environment type")
    P.add_argument("--gpu_type", default="a100", choices=["a100", "v100l"],
        help="GPU type")
    P.add_argument("--mem", default="100G",
        help="RAM—specify SLURM argument '100G'")
    return P.parse_known_args()

if __name__ == "__main__":
    slurm_args, unparsed_args = get_slurm_args()

    if slurm_args.script == "DAE.py":
        args = DAE.get_args(unparsed_args)
        name = os.path.basename(DAE.dae_model_folder(args))
        args, file_move_command = get_args_with_data_on_node(args,
            arg_names_to_move=["data_tr", "data_val"])
        num_gpus = len(args.gpus)
        num_cpus = min(24, max(1, num_gpus) * 12)
    elif slurm_args.script == "IMLE_DAE.py":
        args = IMLE_DAE.get_args(unparsed_args)
        args, file_move_command = get_args_with_data_on_node(args,
            arg_names_to_move=["data_tr", "data_val"])
        name = os.path.basename(IMLE_DAE.imle_model_folder(args))
        num_gpus = len(args.gpus)
        num_cpus = min(24, max(1, num_gpus) * 12)
    else:
        raise NotImplementedError()

    script = f"{file_move_command}\npython {slurm_args.script} {unparse_args(args)} --job_id $SLURM_JOB_ID --num_workers {num_cpus}"

    if slurm_args.env == "conda":
        env_str = "conda activate py3103MRL"
    elif slurm_args.env == "pip":
        env_str = "module load python/3.10\nsource ~/virtual_envs/py3103MRL/bin/activate"

    with open("slurm/slurm_template.txt", "r") as f:
        slurm_template = f.read()
    slurm_template = slurm_template.replace("ACCOUNT", slurm_args.account)
    slurm_template = slurm_template.replace("TIME", slurm_args.time)
    slurm_template = slurm_template.replace("NUM_GPUS", str(num_gpus))
    slurm_template = slurm_template.replace("NUM_CPUS", str(num_cpus))
    slurm_template = slurm_template.replace("NAME", name)
    slurm_template = slurm_template.replace("PYTHON_ENV_STR", env_str)
    slurm_template = slurm_template.replace("SCRIPT", script)
    slurm_template = slurm_template.replace("GPU_TYPE", slurm_args.gpu_type)
    slurm_template = slurm_template.replace("MEM", slurm_args.mem)

    slurm_script = f"slurm/{name}.sh"
    with open(slurm_script, "w+") as f:
        f.write(slurm_template)

    tqdm.write(f"File move command: {file_move_command}")
    tqdm.write(f"Script:\n{script}")
    tqdm.write(f"SLURM submission script written to {slurm_script}")
    tqdm.write(f"Outputs will write to job_results/{name}.txt")
    os.system(f"sbatch {slurm_script}")


        




        