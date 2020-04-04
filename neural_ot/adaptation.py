import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from framework import *

torch.manual_seed(1)

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