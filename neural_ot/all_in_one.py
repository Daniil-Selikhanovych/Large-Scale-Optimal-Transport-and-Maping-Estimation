import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.datasets as dset
from PIL import Image
import matplotlib.pyplot as plt

try:
    device = torch.device("cuda:1")
    is_cuda = True
except:
    device = torch.device("cpu")
    is_cuda = False

torch.manual_seed(1)

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

h, w = 16, 16
batch_size = 1000

tr = transforms.Compose([transforms.Resize((h, w)),
                         transforms.ToTensor()])

mnist = dset.MNIST("data/mnist", download=True, transform=tr)
usps = dset.USPS("data/usps", download=True, transform=transforms.ToTensor())

n_batches = max(len(mnist), len(usps))//batch_size + 1
pairs_loader = ZipLoader([mnist, usps], batch_size=batch_size, n_batches=n_batches, 
                          pin_memory=is_cuda, return_idx=True, 
                          num_workers=8)

u = Vector(len(mnist))
torch.nn.init.normal_(u.v, 15, 6)
# torch.nn.init.normal_(u.v, 5, 2)
v = Vector(len(usps))
torch.nn.init.normal_(v.v, 15, 6)
# torch.nn.init.normal_(v.v, 5, 2)

# f = nn.Sequential(Reshape(-1, h*w),
#                   nn.Linear(h*w, 200),
#                   nn.BatchNorm1d(200),
#                   nn.SELU(),
#                   nn.Linear(200, 500),
#                   nn.BatchNorm1d(500),
#                   nn.SELU(),
#                   nn.Linear(500, 1000),
#                   nn.BatchNorm1d(1000),
#                   nn.SELU(),
#                   nn.Linear(1000, h*w),
#                   nn.Sigmoid(),
#                   Reshape(-1, 1, h, w))


f = nn.Sequential(nn.Conv2d(1, 8, 3, stride=1, padding=1), # 16
                  nn.BatchNorm2d(8),
                  nn.SELU(),
                  nn.Conv2d(8, 16, 3, stride=2, padding=1), # 8
                  nn.BatchNorm2d(16),
                  nn.SELU(),
                  nn.Conv2d(16, 32, 3, stride=2, padding=1), # 4
                  nn.BatchNorm2d(32),
                  nn.SELU(),
                  nn.Conv2d(32, 64, 3, stride=2, padding=1), # 2
                  nn.BatchNorm2d(64),
                  nn.SELU(),
                  nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0), # 4
                  nn.BatchNorm2d(32),
                  nn.SELU(),
                  nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1), # 8
                  nn.BatchNorm2d(16),
                  nn.SELU(),
                  nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1), # 16
                  nn.BatchNorm2d(8),
                  nn.SELU(),
                  nn.ConvTranspose2d(8, 1, kernel_size=3, stride=1, padding=1), # 16
                  nn.Sigmoid(),
                  Interpolator(16, 16)
)

regularization_parameter = 0.07
regularization_mode = "l2"
model = NeuralOT(u, v, f, regularization_parameter=regularization_parameter, 
                regularization_mode=regularization_mode,
                from_discrete=True, to_discrete=True)

plan_optimizer = torch.optim.Adam([u.v, v.v], lr=1.)
mapping_optimizer = torch.optim.Adam(f.parameters(), lr=1e-3)
plan_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(plan_optimizer, 
                                                            factor=0.5)
mapping_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mapping_optimizer, 
                                                            factor=0.5)

print("Training u, v")   
n_epochs = 200                               
train(n_epochs, model, model.plan_criterion, plan_optimizer, pairs_loader,
    scheduler=plan_scheduler)
torch.save(model.cpu(), "model_last")
print("Training f")
train(n_epochs//2, model, model.mapping_criterion, mapping_optimizer, pairs_loader,
    scheduler=mapping_scheduler)
torch.save(model.cpu(), f"model_last")

n_samples = 5
model = torch.load("model_last")

fig, axes = plt.subplots(2, n_samples, figsize=(10, 6))

for i in range(n_samples):
    img = mnist[i][0]
    axes[0, i].imshow(img.squeeze(), cmap="Greys")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    mapped = model.map(img)
    axes[1, i].imshow(mapped.squeeze().detach().numpy(), cmap="Greys")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    
plt.tight_layout()
fig.savefig("last.png")