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
        
def evaluate(model, loader_te, args, noise=False):
    """
    """
    correct, total, loss_te = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    with torch.no_grad():
        for x,y in loader_te:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = x.view(x.shape[0], -1)
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
    model = ProbeWithBackbone(model.encoder).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.probe.parameters(),
        lr=args.probe_lr,
        weight_decay=1e-5)
    
    for e in tqdm(range(args.probe_epochs),
        desc="Probe Epochs",
        leave=False,
        dynamic_ncols=True):

        for x,y in loader_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x = x.view(x.shape[0], -1)
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return evaluate(model, loader_te, args)

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
            x = x.view(x.shape[0], -1)
            nx = Utils.with_noise(x)
            loss = loss_fn(probe(nx), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return evaluate(probe, loader_te, args, noise=True)    

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
            x = x.view(x.shape[0], -1)
            loss = loss_fn(probe(x), y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    return evaluate(probe, loader_te, args, noise=False) 

