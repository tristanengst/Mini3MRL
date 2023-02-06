from functools import partial
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

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
    def __init__(self, encoder, num_classes=10):
        super(ProbeWithBackbone, self).__init__()
        self.encoder = encoder
        self.probe = nn.Linear(encoder.feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            fx = self.encoder(x)
        return self.probe(fx)

class MLPProbeWithBackbone(ProbeWithBackbone):
    """Neural network implementing a learnable MLP probe over a frozen
    encoder [encoder].

    Args:
    encoder     -- encoder behind probe
    num_classes -- number of classes to predict
    """
    def __init__(self, encoder, num_classes=10):
        super(MLPProbeWithBackbone, self).__init__(encoder)
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
    model = ProbeWithBackbone(backbone).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.probe.parameters(),
        lr=args.probe_lrs[1],
        weight_decay=1e-5)
    scheduler = Utils.StepScheduler(optimizer, args.probe_lrs)
    
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

        if args.probe_verbosity > 0 and e % (args.probe_epochs // 6) == 0:
            loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
            loss_val, acc_val = evaluate_probe(model, loader_val, args)
            tqdm.write(f"\tLinear probe epoch {e:4}/{args.probe_epochs} - loss/tr={loss_tr} loss/val={loss_val} acc/tr={acc_tr} acc/val={acc_val}")

    loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
    loss_val, acc_val = evaluate_probe(model, loader_val, args)
    return {"acc/linear_probe/tr": acc_tr,
        "acc/linear_probe/val": acc_val,
        "loss/linear_probe/tr": loss_tr,
        "loss/linear_probe/val": loss_val}

    return accuracy(model, loader_val, args)

def mlp_probe(model, loader_tr, loader_val, args, **kwargs):
    """Returns the accuracy on [loader_val] of an MLP probe on the
    representations of the encoder of [model] run on [loader_tr].
    """
    backbone = get_encoder_from_model(model, **kwargs)
    model = MLPProbeWithBackbone(backbone).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.probe.parameters(),
        lr=args.probe_lrs[1],
        weight_decay=1e-5)
    scheduler = Utils.StepScheduler(optimizer, args.probe_lrs)
    
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

        if args.probe_verbosity > 0 and e % (args.probe_epochs // 6) == 0:
            loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
            loss_val, acc_val = evaluate_probe(model, loader_val, args)
            tqdm.write(f"\tMLP probe epoch {e:4}/{args.probe_epochs} - loss/tr={loss_tr} loss/val={loss_val} acc/tr={acc_tr} acc/val={acc_val}")

    loss_tr, acc_tr = evaluate_probe(model, loader_tr, args)
    loss_val, acc_val = evaluate_probe(model, loader_val, args)
    return {"acc/mlp_probe/tr": acc_tr,
        "acc/mlp_probe/val": acc_val,
        "loss/mlp_probe/tr": loss_tr,
        "loss/mlp_probe/val": loss_val}


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
