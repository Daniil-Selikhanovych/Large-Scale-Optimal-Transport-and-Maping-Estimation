import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from tqdm import tqdm


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
            idx = [torch.multinomial(w, self.batch_size, replacement=True) for w in self.weights]
            yield torch.stack(idx, dim=1)

    def __len__(self):
        return self.batch_size * self.n_batches


class ZipDataset(Dataset):
    """
    ZipDataset represents a dataset that stores several other datasets zipped together.
    """
    def __init__(self, *datasets, return_targets=False, return_idx=True):
        super().__init__()
        self.datasets = datasets
        self.return_targets = return_targets
        self.return_idx = return_idx

    def __getitem__(self, idx):
        items = []
        for i, ds in zip(idx, self.datasets):
            cur_items = []
            if self.return_idx:
                cur_items.append(i)
            cur_items.append(ds[i][0])
            if self.return_targets:
                cur_items.append(ds[i][1])
            items.append(cur_items)

        if len(items) == 1:
            items = items[0]
        
        return items 

    def __len__(self):
        return np.prod([len(ds) for ds in self.datasets])


class ZipLoader(DataLoader):
    def __init__(self, *datasets, batch_size, n_batches, return_targets=False, return_idx=True, **kwargs):
        """
        ZipLoader allows to sample batches from zipped datasets with possibly different number of elements.
        """
        us = UniformSampler(*datasets, batch_size=batch_size, n_batches=n_batches)
        dl = ZipDataset(*datasets, return_targets=return_targets, return_idx=return_idx)
        super().__init__(dl, batch_sampler=us, **kwargs)


def get_mean_covariance(mnist):
    def rescale(data):
        return 2 * (data / 255 - .5)

    if hasattr(mnist, 'data'):
        rescaled_data = rescale(mnist.data)
    elif hasattr(mnist, 'datasets'):
        rescaled_data = torch.cat([rescale(ds.data) for ds in mnist.datasets])
    else:
        raise ValueError('Argument ``mnist`` is invalid.')

    rescaled_data = rescaled_data.reshape(len(rescaled_data), -1)
    return torch.mean(rescaled_data, 0), torch.from_numpy(np.cov(rescaled_data.T).astype(np.float32))


def gaussian_sampler(mean, covariance, batch_size, n_batches, min_eigval=1e-3):
    eigval, eigvec = torch.symeig(covariance, eigenvectors=True)
    eigval, eigvec = eigval[eigval > min_eigval], eigvec[:, eigval > min_eigval]
    height = width = int(np.sqrt(len(mean)))

    for i in range(n_batches):
        samples = torch.randn(batch_size, len(eigval))
        samples = mean + (torch.sqrt(eigval) * samples) @ eigvec.T
        yield None, samples.reshape(-1, 1, height, width)


class DistributionDataset():
    def __init__(self, distribution, transform=lambda x: x):
        super().__init__()
        self.distribution = distribution
        self.transform = transform
        
    def __getitem__(self, idx):
        return self.transform(self.distribution.sample()), None
    
    def __len__(self):
        return 1


def get_rotation(theta):
    rad = np.radians(theta)
    c, s = np.cos(rad), np.sin(rad)
    R = np.array([[c, -s],
                  [s,  c]])
    return R


class CircleDataset():
    def __init__(self, n_samples, n_centers=9, sigma=0.02):
        super().__init__()
        self.nus = [torch.zeros(2)]
        self.sigma = sigma
        for i in range(n_centers-1):
            R = get_rotation(i*360/(n_centers-1))
            self.nus.append(torch.tensor([1, 0] @ R, dtype=torch.float))
        classes = torch.multinomial(torch.ones(n_centers), n_samples, 
                                    replacement=True)
        
        data = []
        for i in range(n_centers):
            n_samples_class = torch.sum(classes == i)
            if n_samples_class == 0:
                continue
            dist = MultivariateNormal(self.nus[i], 
                                      torch.eye(2)*self.sigma**2)
            data.append(dist.sample([n_samples_class.item()]))
        self.data = torch.cat(data)
        
    def __getitem__(self, idx):
        return self.data[idx], None
    
    def __len__(self):
        return self.data.shape[0]


class CustomGaussian:
    def __init__(self, mean, covariance, min_eigval=1e-3):
        self.mean = mean
        eigval, eigvec = torch.symeig(covariance, eigenvectors=True)
        self.eigval, self.eigvec = eigval[eigval > min_eigval], eigvec[:, eigval > min_eigval]
        self.height = self.width = int(np.sqrt(len(mean))) 

    def sample(self):
        x = torch.randn(1, len(self.eigval))
        x = self.mean + (torch.sqrt(self.eigval) * x) @ self.eigvec.T
        return x