from tqdm import tqdm
import torch
import torch.nn as nn

import Utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProbeWithBackbone(nn.Module):
    """
    """
    def __init__(self, encoder):
        super(ProbeWithBackbone, self).__init__()
        self.encoder = encoder
        self.probe = nn.Linear(64, 10)

    def forward(self, x):
        with torch.no_grad():
            fx = self.encoder(x)
        return self.probe(fx)
        
def accuracy(model, loader_te, args, noise=False):
    """Returns the accuracy of [model] in classifying data from [loader_te]."""
    correct, total, loss_te = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        for x,y in loader_te:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = Utils.with_noise(x) if noise else x
            fx = model(x)
            loss = loss_fn(fx, y)

            loss_te += loss * len(x)
            correct += torch.sum(torch.argmax(fx, dim=1) == y)
            total += len(x)
    
    return correct.item() / total

def linear_probe(model, loader_tr, loader_te, args):
    """
    """
    model = ProbeWithBackbone(Utils.de_dataparallel(model).encoder).to(device)
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

    return accuracy(model, loader_te, args)

def noised_linear_probe(loader_tr, loader_te, args):
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

    return accuracy(probe, loader_te, args, noise=True)    

def plain_linear_probe(loader_tr, loader_te, args):
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

    return accuracy(probe, loader_te, args, noise=False) 

