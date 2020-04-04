import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class UniformSampler:
    """
    UniformSampler allows to sample batches in random manner without splitting the original data.
    """
    def __init__(self, *datasets, batch_size=1, n_batches=1):
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.weights = [torch.ones(len(ds)) for ds in datasets]

    def __iter__(self):
        for i in range(self.n_batches):
            idx = [torch.multinomial(w, self.batch_size) for w in self.weights]
            yield torch.stack(idx, dim=1)

    def __len__(self):
        return self.batch_size * self.n_batches


class ZipDataset(Dataset):
    """
    ZipDataset represents a dataset that stores several other datasets zipped together.
    """
    def __init__(self, *datasets, return_targets=False):
        super().__init__()
        self.datasets = datasets
        self.return_targets = return_targets

    def __getitem__(self, idx):
        items = []
        for i, ds in zip(idx, self.datasets):
            if self.return_targets:
                items.append(ds[i])
            else:
                items.append(ds[i][0])
        
        if len(items) == 1:
            items = items[0]
        
        return items 

    def __len__(self):
        return np.prod([len(ds) for ds in self.datasets])


class ZipLoader(DataLoader):
    def __init__(self, *datasets, batch_size, n_batches, return_targets=False, **kwargs):
        """
        ZipLoader allows to sample batches from zipped datasets with possibly different number of elements.
        """
        us = UniformSampler(*datasets, batch_size=batch_size, n_batches=n_batches)
        dl = ZipDataset(*datasets, return_targets=return_targets)
        super().__init__(dl, batch_sampler=us, **kwargs)
