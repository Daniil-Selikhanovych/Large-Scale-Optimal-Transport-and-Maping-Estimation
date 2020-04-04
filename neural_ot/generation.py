import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from framework import *
from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(1)
h, w = 28, 28
batch_size = 1000

mnist = dset.MNIST("data/mnist", download=True, transform=transforms.ToTensor())
X = mnist.data.reshape(-1, h*w).float()/255
mu = torch.mean(X, dim=0, dtype=torch.float)
sigma = torch.tensor(np.cov(X.T), dtype=torch.float)

eps = 1e-5
distr = MultivariateNormal(mu, sigma+eps*torch.eye(h*w))
distr_dset = DistributionDataset(distr, transform=lambda x: x.reshape(1, h, w))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

x_new = distr.sample().reshape(28, 28)
ax.imshow(x_new, cmap="Greys")
fig.savefig("sample.png")

n_batches = len(mnist)//batch_size + 1

pairs_loader = ZipLoader([distr_dset, mnist], batch_size=batch_size, n_batches=n_batches, 
                          pin_memory=is_cuda, return_idx=True, num_workers=8)

u = nn.Sequential(Reshape(-1, h*w),
                  nn.Linear(h*w, 1024),
                  nn.BatchNorm1d(1024),
                  nn.SELU(),
                  nn.Linear(1024, 1024),
                  nn.BatchNorm1d(1024),
                  nn.SELU(),
                  nn.Linear(1024, 1))
v = Vector(len(mnist))
torch.nn.init.normal_(v.v, 15, 6)

f = nn.Sequential(Reshape(-1, h*w),
                  nn.Linear(h*w, 1024),
                  nn.BatchNorm1d(1024),
                  nn.SELU(),
                  nn.Linear(1024, 1024),
                  nn.BatchNorm1d(1024),
                  nn.SELU(),
                  nn.Linear(1024, h*w),
                  nn.Sigmoid(),
                  Reshape(-1, 1, h, w))

regularization_parameter = 0.07
regularization_mode = "l2"
model = NeuralOT(u, v, f, regularization_parameter=regularization_parameter, 
                regularization_mode=regularization_mode,
                from_discrete=False, to_discrete=True)

plan_optimizer = torch.optim.Adam(list(u.parameters())+[v.v], lr=1e-4)
mapping_optimizer = torch.optim.Adam(f.parameters(), lr=1e-4)
plan_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(plan_optimizer, 
                                                            factor=0.5)
mapping_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(mapping_optimizer, 
                                                            factor=0.5)

print("Training u, v")   
n_epochs = 200                              
train(n_epochs, model, model.plan_criterion, plan_optimizer, pairs_loader,
    scheduler=plan_scheduler)
torch.save(model.cpu(), "generator")
print("Training f")
train(n_epochs//2, model, model.mapping_criterion, mapping_optimizer, pairs_loader,
    scheduler=mapping_scheduler)
torch.save(model.cpu(), f"generator")

n_samples = 5
model = torch.load("generator")

fig, axes = plt.subplots(2, n_samples, figsize=(10, 6))

for i in range(n_samples):
    img = distr.sample()
    axes[0, i].imshow(img.reshape(h, w), cmap="Greys")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    mapped = model.map(img)
    axes[1, i].imshow(mapped.squeeze().detach().numpy(), cmap="Greys")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    
plt.tight_layout()
fig.savefig("generated.png")