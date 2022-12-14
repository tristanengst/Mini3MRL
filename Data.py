from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10
from tqdm import tqdm
import Utils

def get_dataset(s, transform=None, split=None):
    """Returns the dataset given by string [s], a path to an ImageFolder or one
    of 'mnist' or 'cifar10'. In these two cases, [split] determines what split
    of the data is returned.
    """
    train = (not split == "test")
    if s == "cifar10":
        return CIFAR10(root="cifar10", train=train, download=True, transform=transform)
    elif s == "mnist":
        return MNIST(root="mnist", train=train, download=True, transform=transform)
    else:
        return ImageFolder(s, transform=transform)

def get_fewshot_dataset(dataset, n_way=-1, n_shot=-1, classes=None, seed=0,   
    fewer_shots_if_needed=False):
    """Returns a Subset of [dataset] giving a n-shot n-way task.

    Args:
    dataset             -- ImageFolder-like dataset
    n_way               -- number of classes to use, -1 for all available
    n_shot              -- number of shots to use, -1 for all available
    classes             -- classes to use (overrides [n_way])
    allow_fewer_shots   -- if [dataset] doesn't have all the [n_shots] for a
                                class, use less than [n_shots]
    """
    if classes == -1 and n_shot == -1:
        return dataset
    else:
        if classes == -1:
            chosen_classes = set(dataset.classes)
        elif classes is None:
            n_way = len(dataset.classes) if n_way == -1 else n_way
            chosen_classes = set(Utils.sample(dataset.classes, k=n_way, seed=seed))
        else:
            chosen_classes = set(classes)
        
        chosen_targets = {dataset.class_to_idx[c] for c in chosen_classes}
        class2idxs = defaultdict(lambda: [])

        targets = [int(t) for t in dataset.targets]
        for idx,t in enumerate(targets):
            if t in chosen_targets:
                class2idxs[t].append(idx)
        
        if n_shot == -1:
            pass
        else:
            n_shot_fn = lambda x: (min(len(x), n_shot) if fewer_shots_if_needed else n_shot)
            try:
                class2idxs = {c: Utils.sample(idxs, k=n_shot_fn(idxs), seed=seed)
                    for c,idxs in class2idxs.items()}
            except ValueError as e:
                class2n_idxs = "\n".join([f"\t{c}: {len(idxs)}"
                    for c,idxs in class2idxs.items()])
                tqdm.write(f"Likely --val_n_shot asked for more examples than are available | val_n_shot {n_shot} | class to num idxs: {class2n_idxs}")
                raise e
    
        indices = Utils.flatten([idxs for idxs in class2idxs.values()])
        return ImageFolderSubset(dataset, indices=indices)

class ImageFolderSubset(Dataset):
    """Subset of an ImageFolder that preserves key attributes. Besides
    preserving ImageFolder-like attributes, the key improvement over a regular
    Subset is a target2idx dictionary that maps a target returned from [data] to
    a number in [0, len(classes)) which is necessary for classification.

    ImageFolders have the following key attributes:
    - classes : a list of the classes in the ImageFolder
    - class_to_idx : a mapping from [classes] to an index in [0, len(classes))
    - root : place from which the data is loaded
    - samples : a list of (x,y) tuples comprising the data
    - targets : a list where the ith element is the target of the ith example

    When constructing a subset of an ImageFolder, we need to reconstruct these
    attributes given the data in the subset. This isn't done by the regular
    Subset class, and can't be accomplished by just subsetting the ImageFolder's
    data. For instance, if we construct a subset of Imagenet with only the last
    three classes, the targets need to be indexed 0, 1, and 2.

    Args:
    data    -- ImageFolder-like dataset 
    indices -- list giving subset indices to [data] to comprise the subset
    """
    def __init__(self, data, indices):
        super(ImageFolderSubset, self).__init__()
        self.indices = list(indices)
        self.data = data

        if isinstance(data.targets, list):
            data_targets = np.array(data.targets)
        elif isinstance(data.targets, torch.Tensor):
            data_targets = data.targets.numpy()

        idxs = np.array(indices).astype(np.int64)
        superset_targets = list(data_targets[idxs])
        superset_target_set = set(superset_targets)
        superset_class2idx = {c: int(t) for c,t in data.class_to_idx.items() if t in superset_target_set}

        self.target2idx = {int(t): idx for idx,t in enumerate(sorted(superset_target_set))}

        self.classes = list(superset_class2idx.keys())
        self.class2idx = {c: self.target2idx[t] for c,t in superset_class2idx.items()}
        self.root = data.root

        # Correct ImageFolders have a 'samples' attribute, and this is what we
        # build. However, some datasets *glares at the MNIST and CIFAR10*
        # contain a 'data' attribute that stores just the x-values.
        if hasattr(data, "samples"):
            self.samples = [(data.samples[idx][0], self.target2idx[data.samples[idx][1]]) for idx in indices]
        elif hasattr(data, "data"):
            self.samples = [(data.data[idx], self.target2idx[int(data.targets[idx])]) for idx in indices]
        else:
            raise NotImplementedError()
       
        self.targets = [y for _,y in self.samples]

        self.transform = data.transform
        self.target_transform = data.target_transform
        self.loader = data.loader if hasattr(data, "loader") else (lambda x: x)

    def __str__(self): return f"{self.__class__.__name__} [root={self.root} length={self.__len__()}]"

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
    
    @staticmethod
    def complement(data, same_classes=True):
        """Returns an ImageFolderSubset containing the classes of [data] but
        all the unused indices to data in these classes.
        """
        if same_classes:
            used_idxs = set(data.indices)
            unused_idxs = [idx for idx,t in enumerate(data.data.targets)
                if int(t) in data.target2idx and not idx in used_idxs]
            return ImageFolderSubset(data.data, indices=unused_idxs)
        else:
            raise NotImplementedError()

        