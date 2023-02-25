import argparse
from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import IO
import Data
import Models
import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def probe(model, loader_tr, loader_val, args):
    """Returns a dictionary giving the test accuracies of various kinds of
    linear probes given by [args].

    Args:
    model       -- Complete model used in SSL
    loader_tr   -- DataLoader over training data for the probe
    loader_val   -- DataLoader over validation data for the probe
    args        -- argparse Namespace with experiment parameters
    """
    eval_fns = []

    if args.probe_linear:
        if args.probe_include_codes in [1, 2]:
            eval_fns.append(partial(linear_probe, include_codes=True))
        elif args.probe_include_codes in [0, 2]:
            eval_fns.append(partial(linear_probe, include_codes=False))
    if args.probe_mlp:
        if args.probe_include_codes in [1, 2]:
            eval_fns.append(partial(mlp_probe, include_codes=True))
        elif args.probe_include_codes in [0, 2]:
            eval_fns.append(partial(mlp_probe, include_codes=False))

    results = [f(model, loader_tr, loader_val, args) for f in tqdm(eval_fns,
        desc="Running probes",
        leave=False,
        dynamic_ncols=True)]
    return {k: v for r in results for k,v in r.items()}
    

def get_encoder_from_model(model, include_codes=False):
    """Returns the enocder from [model] set to include codes as per [include_codes]."""
    if include_codes:
        return model.to_encoder_with_ada_in(Utils.de_dataparallel(model))
    else:
        return Utils.de_dataparallel(model).encoder


class ProbeWithBackbone(nn.Module):
    """Neural network implementing a learnable linear probe over a frozen
    encoder [encoder].

    Args:
    encoder     -- encoder behind probe
    num_classes -- number of classes to predict
    """
    def __init__(self, encoder, probe_normalize_feats=False, num_classes=10, **kwargs):
        super(ProbeWithBackbone, self).__init__()
        self.encoder = encoder
        self.probe = nn.Linear(encoder.feat_dim, num_classes)
        self.probe_normalize_feats = probe_normalize_feats

    def forward(self, x):
        with torch.no_grad():
            fx = self.encoder(x)
            fx = nn.functional.normalize(fx) if self.probe_normalize_feats else fx

        return self.probe(fx)

class MLPProbeWithBackbone(ProbeWithBackbone):
    """Neural network implementing a learnable MLP probe over a frozen
    encoder [encoder].

    Args:
    encoder     -- encoder behind probe
    num_classes -- number of classes to predict
    """
    def __init__(self, encoder, num_classes=10, **kwargs):
        super(MLPProbeWithBackbone, self).__init__(encoder, **kwargs)
        self.probe = torchvision.ops.MLP(encoder.feat_dim, [512, 512, num_classes])
        
def evaluate_probe(model, loader, args, noise=False):
    """Returns the loss and accuracy of [model] on data from [loader]."""
    correct, total, loss = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = Utils.with_noise(x) if noise else x
            fx = model(x)
            
            loss += loss_fn(fx, y)
            correct += torch.sum(torch.argmax(fx, dim=1) == y)
            total += len(x)
    
    return loss.item() / total, correct.item() / total

def linear_probe(model, loader_tr, loader_val, args, **kwargs):
    """Returns the accuracy on [loader_val] of a linear probe on the
    representations of the encoder of [model] run on [loader_tr].
    """
    backbone = get_encoder_from_model(model, **kwargs)
    model = ProbeWithBackbone(backbone, **vars(args)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.probe.parameters(),
        lr=args.probe_lrs[1],
        weight_decay=1e-5)
    scheduler = Utils.StepScheduler(optimizer, args.probe_lrs)
    
    epoch2accs_tr, epoch2accs_val = {}, {}
    losses_tr, losses_val = [], []
    grad_norms = []
    for e in tqdm(range(args.probe_epochs),
        desc="Probe Epochs",
        leave=False,
        dynamic_ncols=True):

        for x,y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

            grad_norms.append(torch.linalg.norm(model.probe.weight.grad).item())
            
        scheduler.step()

        if e % args.probe_eval_iter == 0 or e == args.probe_epochs - 1:
            loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
            loss_val, acc_val = evaluate_probe(model, loader_val, args)
            epoch2accs_tr[e] = acc_tr
            epoch2accs_val[e] =  acc_val
            losses_tr.append(loss_tr)
            losses_val.append(loss_val)

    if args.probe_verbosity > 0:
        eval_str = f"\tLinear probe epoch to accuracy (train/val): "
        eval_str += " ".join([f"{e}=({tr:.3f}/{val:.3f})" for (e,tr),(_,val) in zip(epoch2accs_tr.items(), epoch2accs_val.items())])
        tqdm.write(eval_str)

    # This measures if the probe broke. We log the delta between the accuracy at
    # iteration [t] and the maximum it was prior to that iteration. If
    # accuracies are monotonically increasing, this is zero. Small negative
    # deltas are okay; large ones aren't.(
    accs_tr = torch.tensor([epoch2accs_tr[k] for k in sorted(epoch2accs_tr)])
    max_up_to_idx = torch.tensor([torch.max(accs_tr[:idx]) for idx in range(1, len(accs_tr)+1)])
    delta_from_max_tr = torch.min(accs_tr - max_up_to_idx)

    accs_val = torch.tensor([epoch2accs_val[k] for k in sorted(epoch2accs_val)])
    max_up_to_idx = torch.tensor([torch.max(accs_val[:idx]) for idx in range(1, len(accs_val)+1)])
    delta_from_max_val = torch.min(accs_val - max_up_to_idx)

    losses_tr = torch.tensor(losses_tr)
    losses_val = torch.tensor(losses_val)

    grad_norms = torch.tensor(grad_norms)

    return {"acc/linear_probe_max/tr": torch.max(accs_tr),
        "acc/linear_probe_start/tr": accs_tr[0],
        "acc/linear_probe_end/tr": accs_tr[-1],
        "acc/linear_probe_max_delta_from_prior_max/tr": delta_from_max_tr,
        "acc/linear_probe_max/val": torch.max(accs_val),
        "acc/linear_probe_start/val": accs_val[0],
        "acc/linear_probe_end/val": accs_val[-1],
        "loss/linear_probe_std/tr": torch.std(losses_tr),
        "loss/linear_probe_std/val": torch.std(losses_val),
        "probe_grads/linear_mean": torch.mean(grad_norms),
        "probe_grads/linear_std": torch.std(grad_norms),
        "probe_grads/linear_max": torch.max(grad_norms),
        "acc/linear_probe_max_delta_from_prior_max/val": delta_from_max_val}

def mlp_probe(model, loader_tr, loader_val, args, **kwargs):
    """Returns the accuracy on [loader_val] of an MLP probe on the
    representations of the encoder of [model] run on [loader_tr].
    """
    backbone = get_encoder_from_model(model, **kwargs)
    model = MLPProbeWithBackbone(backbone, **vars(args)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.probe.parameters(),
        lr=args.probe_lrs[1],
        weight_decay=1e-5)
    scheduler = Utils.StepScheduler(optimizer, args.probe_lrs)

    epoch2accs_tr, epoch2accs_val = {}, {}
    losses_tr, losses_val = [], []
    for e in tqdm(range(args.probe_epochs),
        desc="Probe Epochs",
        leave=False,
        dynamic_ncols=True):

        for x,y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        if e % args.probe_eval_iter == 0 or e == args.probe_epochs - 1:
            loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
            loss_val, acc_val = evaluate_probe(model, loader_val, args)
            epoch2accs_tr[e] = acc_tr
            epoch2accs_val[e] =  acc_val
            losses_tr.append(loss_tr)
            losses_val.append(loss_val)

    if args.probe_verbosity > 0:
        eval_str = f"\tMLP probe epoch to accuracy (train/val): "
        eval_str += " ".join([f"{e}=({tr:.3f}/{val:.3f})" for (e,tr),(_,val) in zip(epoch2accs_tr.items(), epoch2accs_val.items())])
        tqdm.write(eval_str)

    # This measures if the probe broke. We log the delta between the accuracy at
    # iteration [t] and the maximum it was prior to that iteration. If
    # accuracies are monotonically increasing, this is zero. Small negative
    # deltas are okay; large ones aren't.(
    accs_tr = torch.tensor([epoch2accs_tr[k] for k in sorted(epoch2accs_tr)])
    max_up_to_idx = torch.tensor([torch.max(accs_tr[:idx]) for idx in range(1, len(accs_tr)+1)])
    delta_from_max_tr = torch.min(accs_tr - max_up_to_idx)

    accs_val = torch.tensor([epoch2accs_val[k] for k in sorted(epoch2accs_val)])
    max_up_to_idx = torch.tensor([torch.max(accs_val[:idx]) for idx in range(1, len(accs_val)+1)])
    delta_from_max_val = torch.min(accs_val - max_up_to_idx)

    losses_tr = torch.tensor(losses_tr)
    losses_val = torch.tensor(losses_val)

    return {"acc/mlp_probe_max/tr": torch.max(accs_tr),
        "acc/mlp_probe_start/tr": accs_tr[0],
        "acc/mlp_probe_end/tr": accs_tr[-1],
        "acc/mlp_probe_max_delta_from_prior_max/tr": delta_from_max_tr,
        "acc/mlp_probe_max/val": torch.max(accs_val),
        "acc/mlp_probe_start/val": accs_val[0],
        "acc/mlp_probe_end/val": accs_val[-1],
        "loss/mlp_probe_std/tr": torch.std(losses_tr),
        "loss/mlp_probe_std/val": torch.std(losses_val),
        "acc/mlp_probe_max_delta_from_prior_max/val": delta_from_max_val}


def noised_linear_probe(loader_tr, loader_val, args):
    probe = nn.Linear(784, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(),
        lr=args.probe_lr,
        weight_decay=1e-5)
    
    for e in tqdm(range(args.probe_epochs),
        desc="Probe Epochs",
        leave=False,
        dynamic_ncols=True):

        for x,y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            nx = Utils.with_noise(x)
            loss = loss_fn(probe(nx), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return accuracy(probe, loader_val, args, noise=True)    

def plain_linear_probe(loader_tr, loader_val, args):
    probe = nn.Linear(784, 10).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(),
        lr=args.probe_lr,
        weight_decay=1e-5)
    
    for e in tqdm(range(args.probe_epochs),
        desc="Probe Epochs",
        leave=False,
        dynamic_ncols=True):

        for x,y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            loss = loss_fn(probe(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return accuracy(probe, loader_val, args, noise=False) 

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P = IO.parser_with_default_args(P)
    P = IO.parser_with_data_args(P)
    P = IO.parser_with_logging_args(P)
    P = IO.parser_with_probe_args(P)

    args = P.parse_args()
    args.probe_lrs = Utils.StepScheduler.process_lrs(args.probe_lrs)

    states = torch.load(args.resume)
    Utils.set_seed(states["seeds"])
    args = argparse.Namespace(**vars(states["args"]) | vars(args))
    model = Models.get_model(args, imle=True)
    model.load_state_dict(states["model"], strict=False)
    model = nn.DataParallel(model, device_ids=args.gpus).to(device)
    model = model.to(device)
    last_epoch = states["epoch"]

    data_tr, data_val = Data.get_data_from_args(args)
    loader_tr = DataLoader(data_tr,
        batch_size=args.bs,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True)
    loader_val = DataLoader(data_val,
        batch_size=args.bs,
        num_workers=args.num_workers,
        pin_memory=True)

    tqdm.write(f"USING MODEL {args.resume}")

    if args.probe_linear:
        if args.probe_include_codes in [1, 2]:
            raise NotImplementedError()
        elif args.probe_include_codes in [0, 2]:
            probe_acc_max_val = []
            probe_acc_end_val = []
            for t in tqdm(range(args.probe_trials),
                leave=True,
                dynamic_ncols=True,
                desc="Trials"):

                result = linear_probe(model.module, loader_tr, loader_val, args, include_codes=True)
                probe_acc_max_val.append(result["acc/linear_probe_max/val"])
                probe_acc_end_val.append(result["acc/linear_probe_end/val"])

            probe_acc_end_val = torch.tensor(probe_acc_end_val)
            probe_acc_max_val = torch.tensor(probe_acc_max_val)
            tqdm.write(f"LINEAR PROBE=[acc_end/val_mean={probe_acc_end_val.mean():.5f} acc_end/val_std={probe_acc_end_val.std():.5f} acc_max/val_mean={probe_acc_max_val.mean():.5f} acc_max/val_std={probe_acc_max_val.std():.5f}]")

    if args.probe_mlp:
        if args.probe_include_codes in [1, 2]:
            raise NotImplementedError()
        elif args.probe_include_codes in [0, 2]:
            probe_acc_max_val = []
            probe_acc_end_val = []
            for t in tqdm(range(args.probe_trials),
                leave=True,
                dynamic_ncols=True,
                desc="Trials"):

                result = mlp_probe(model.module, loader_tr, loader_val, args, include_codes=True)
                probe_acc_max_val.append(result["acc/mlp_probe_max/val"])
                probe_acc_end_val.append(result["acc/mlp_probe_end/val"])

            probe_acc_end_val = torch.tensor(probe_acc_end_val)
            probe_acc_max_val = torch.tensor(probe_acc_max_val)
            tqdm.write(f"MLP PROBE=[acc_end/val_mean={probe_acc_end_val.mean():.5f} acc_end/val_std={probe_acc_end_val.std():.5f} acc_max/val_mean={probe_acc_max_val.mean():.5f} acc_max/val_std={probe_acc_max_val.std():.5f}]")





    
