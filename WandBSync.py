"""The latest weapon for WandB on Narval: automated offline job syncing!

USAGE: python WandBSync.py

Notes:
- If you want to remove a run—possibly split into one or more chunks—you must
    do the following in the order presented to avoid breaking WandB:
    1. Stop the run on on ComputeCanada
    2. Run this script
    3. Remove the run on WandB
"""
import os
from tqdm import tqdm
import shutil
from subprocess import Popen, PIPE, STDOUT

def get_sync_result(f):
    tqdm.write(f"--- {f} ---")
    result = {"can_delete_job": False, "finished": False}
    with Popen(f"wandb sync {f}".split(),
        stdout=PIPE,
        stderr=STDOUT,
        bufsize=1,
        universal_newlines=True) as p:
        for line in p.stdout:
            tqdm.write(f"\t{line}")
            if "was previously created and deleted" in line:
                result["can_delete_job"] = True
            elif ".wandb file is empty" in line:
                result["can_delete_job"] = True
            else:
                continue

            # When this is true, we need to end the process and return.
            # Otherwise, we need to let it continue so things can sync.
            if result["can_delete_job"]:
                p.kill()
                return result
        
        p.kill()
        if os.path.exists(f"{f}/files/wandb-summary.json"):
            result["finished"] = True

        return result

lock_file = ".wandb_sync_lock_file"
if os.path.exists(lock_file):
    pass
else:
    with open(lock_file, "w+") as f:
        f.write("Please don't delete me!")
    tqdm.write(f"Created lock file")

    try:
        files = [f"wandb/{f}" for f in os.listdir("wandb") if f.startswith("offline-run")]
        for f in tqdm(files):

            sync_result = get_sync_result(f)
            tqdm.write(str(sync_result))

            if os.path.exists(f"{f}/files/wandb-summary.json"):
                tqdm.write(f"Run completed, synced")
                shutil.rmtree(f)
            elif sync_result["finished"]:
                tqdm.write(f"Run completed, not synced.")
            
            if sync_result["can_delete_job"] and os.path.exists(f):
                tqdm.write(f"Removing job that had an error in syncing...")
                shutil.rmtree(f)
    except Exception as e:
        os.remove(lock_file)
        tqdm.write(f"Removed lock file {lock_file}; raising error")
        raise e
    finally:
        tqdm.write(f"Removed lock file {lock_file}")
        os.remove(lock_file)
    

        
