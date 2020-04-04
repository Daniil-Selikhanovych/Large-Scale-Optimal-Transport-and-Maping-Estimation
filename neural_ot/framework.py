import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    device = torch.device("cuda:1")
    is_cuda = True
except:
    device = torch.device("cpu")
    is_cuda = False

class NeuralOT(nn.Module):
    def __init__(self, source_dual_net, target_dual_net, source_to_target_net, regularization_parameter=1e-2,
                 regularization_mode='entropy', from_discrete=True, to_discrete=True):
        super().__init__()
        self.from_discrete = from_discrete
        self.to_discrete = to_discrete
        self.u = source_dual_net
        self.v = target_dual_net
        self.f = source_to_target_net
        self.eps = regularization_parameter
        if regularization_mode not in ['entropy', 'l2']:
            raise ValueError("``regularization_mode`` must be ``'entropy'`` or ``'l2'``.")
        self.mode = regularization_mode

    @staticmethod
    def l2_distances(x, y):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N_s, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N_t, C, H, W)``.

        Returns
        -------
        distances : torch.tensor
            Tensor of shape ``(N_s, N_t)`` with pairwise l2 distances between source and target images.
        """
        # return torch.sqrt(torch.sum((x - y) ** 2, dim=(-3, -2, -1)))
        return torch.sum((x - y) ** 2, dim=(-3, -2, -1))

    def plan_criterion(self, x, x_idx, y, y_idx):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N_s, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N_t, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training dual neural networks.
        """
        self.u.train()
        self.v.train()
        if self.from_discrete:
            u = self.u(x_idx)
        else:
            u = self.u(x)
            
        if self.to_discrete:
            v = self.v(y_idx)
        else:
            v = self.v(y)
        c = self.l2_distances(x, y)

        if self.mode == 'entropy':
            regularization_term = -self.eps * torch.exp((u + v - c) / self.eps)
        else:
            regularization_term = -torch.relu(u + v - c) ** 2 / (4 * self.eps)

        # print("Satisfied: {:.0%}".format(torch.mean(u + v < c, dtype=torch.float)))

        return -torch.mean(u + v + regularization_term)

    def mapping_criterion(self, x, x_idx, y, y_idx):
        """
        Parameters
        ----------
        x : torch.tensor
            Batch of images from source domain of shape ``(N, C, H, W)``.
        y : torch.tensor
            Batch of images from target domain of shape ``(N, C, H, W)``.

        Returns
        -------
        loss : torch.tensor
            Loss for training mapping neural network.
        """
        self.u.eval()
        self.v.eval()
        self.f.train()
        with torch.no_grad():
            if self.from_discrete:
                u = self.u(x_idx)
            else:
                u = self.u(x)
                
            if self.to_discrete:
                v = self.v(y_idx)
            else:
                v = self.v(y)
                
        c = self.l2_distances(x, y)
        mapped = self.f(x)  # shape ``(N, C, H, W)``

        d = self.l2_distances(mapped, y)
        if self.mode == 'entropy':
            h = torch.exp((u + v - c) / self.eps)
        else:
            h = torch.relu(u + v - c) / (2 * self.eps)
        return torch.mean(d * h)

    def map(self, x):
        self.f.eval()
        return self.f(x)

def train(epochs, model, criterion, optimizer, train_loader, val_loader=None, scheduler=None, verbose=True, save_dir=None):
    model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        loss_avg = 0
        for batch in train_loader:
            for i in range(len(batch)):
                batch[i] = batch[i].to(device)
            optimizer.zero_grad()
            loss = criterion(*batch)
            loss_avg += loss.item()
            loss.backward()
            optimizer.step()
        loss_avg /= len(train_loader)
        
        model.eval()
        val_loss = 0
        if val_loader:
            with torch.no_grad():
                for batch in val_loader:
                    for i in range(len(batch)):
                        batch[i] = batch[i].to(device)
                    val_loss += criterion(*batch)
            val_loss /= len(val_loader)
         
            if scheduler is not None:
                scheduler.step(val_loss)
        else:
            if scheduler is not None:
                scheduler.step(loss_avg)
        freq = max(epochs//20,1)
        if verbose and epoch%freq==0:
            if val_loader:
                print('Epoch {}/{} || Loss:  Train {:.6g} | Validation {:.6g}'.format(epoch, epochs, loss_avg, val_loss.item()))
            else:
                print('Epoch {}/{} || Loss:  Train {:.6g}'.format(epoch, epochs, loss_avg))

class UniformSampler:
    """
    UniformSampler allows to sample batches in random manner without splitting the original data.
    """
    def __init__(self, datasets, batch_size=1, n_batches=1):
        if not isinstance(datasets, list):
            datasets = [datasets]
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.weights = [torch.ones(len(dset)) for dset in datasets]

    def __iter__(self):
        for i in range(self.n_batches):
            idx = [torch.multinomial(w, self.batch_size) for w in self.weights]
            yield torch.stack(idx, dim=1).squeeze()

    def __len__(self):
        return self.batch_size*self.n_batches

class ZipDataset(Dataset):
    """
    ZipDataset represents a dataset that stores several other datasets zipped together.
    """
    def __init__(self, datasets, return_idx=True):
        super().__init__()
        self.datasets = datasets
        self.return_idx = return_idx

    def __getitem__(self, idx):
        items = []
        for ids, dset in zip(idx, self.datasets):
            items.append(dset[ids][0])
            if self.return_idx:
                items.append(ids)
        
        if len(items) == 1:
            items = items[0]
        
        return items 

    def __len__(self):
        return max([len(dset) for dset in self.datasets])

class ZipLoader(DataLoader):
    def __init__(self, datasets, batch_size, n_batches, *args, return_idx=True, **kwargs):
        """
        ZipLoader allows to sample batches from zipped datasets with possibly different number of elements.
        """
        us = UniformSampler(datasets, batch_size=batch_size, n_batches=n_batches)
        dl = ZipDataset(datasets, return_idx=return_idx)
        self.size = max([len(dset) for dset in datasets])
        super().__init__(dl, *args, batch_sampler=us, **kwargs)
        
    def __len__(self):
        return self.size

class Vector(nn.Module):
    def __init__(self, n_dims):
        super().__init__()
        self.v = nn.Parameter(torch.zeros(n_dims))
        nn.init.xavier_uniform_(self.v[None, :])
    def forward(self, idx):
        return self.v[idx]

class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        
    def forward(self, input):
        return input.view(*self.shape)

class Debugger(nn.Module):
    def __init__(self, name=""):
        super().__init__()
        self.name = name
        
    def forward(self, input):
        print(self.name, input.shape)
        return input

class Interpolator(nn.Module):
    def __init__(self, *shape, mode="bilinear"):
        super().__init__()
        self.shape = shape
        self.mode = mode
        
    def forward(self, input):
        return torch.nn.functional.interpolate(input, self.shape, mode=self.mode,
                                               align_corners=False)