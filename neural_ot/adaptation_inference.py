import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as dset

import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

from framework import *

h, w = 16, 16
tr = transforms.Compose([transforms.Resize((h, w)),
                         transforms.ToTensor()])
mnist = dset.MNIST("data/mnist", download=True, transform=tr)
usps = dset.USPS("data/usps", download=True, transform=transforms.ToTensor())

n_samples = 10
idx = torch.multinomial(torch.ones(len(mnist)), n_samples)
model = torch.load("model_last")
model.eval()

fig, axes = plt.subplots(2, n_samples, figsize=(20, 6))

for i in range(n_samples):
    img = mnist[idx[i]][0]
    axes[0, i].imshow(img.squeeze(), cmap="Greys")
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])

    mapped = model.map(img.reshape(1, 1, h, w))
    axes[1, i].imshow(mapped.squeeze().detach().numpy(), cmap="Greys")
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    
plt.tight_layout()
fig.savefig("last.png")

X_source, y_source = [], []
for i in tqdm(range(len(mnist)), "MNIST"):
    X, y = mnist[i]
    X_source.append(X)
    y_source.append(y)

X_source = torch.cat(X_source).reshape(-1, h*w).numpy()
y_source = np.array(y_source)

X_target, y_target = [], []
for i in tqdm(range(len(usps)), "USPS"):
    X, y = usps[i]
    X_target.append(X)
    y_target.append(y)

X_target = torch.cat(X_target).reshape(-1, h*w).numpy()
y_target = np.array(y_target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

clf = KNeighborsClassifier(n_neighbors=1, metric="correlation")
clf.fit(X_source, y_source)

y_pred = clf.predict(X_target)
print("1-KNN accuracy:", accuracy_score(y_target, y_pred))

X_source_mapped, y_source_mapped = [], []
for i in tqdm(range(len(mnist)), "MNIST -> USPS"):
    X, y = mnist[i]
    mapped = model.map(X.reshape(1, 1, h, w))
    X_source_mapped.append(mapped.squeeze())
    y_source_mapped.append(y)

X_source_mapped = torch.cat(X_source_mapped).reshape(-1, h*w).detach().numpy()
y_source_mapped = np.array(y_source_mapped)

clf = KNeighborsClassifier(n_neighbors=1, metric="correlation")
clf.fit(X_source_mapped, y_source_mapped)

y_pred = clf.predict(X_target)
print("Mapped 1-KNN accuracy:", accuracy_score(y_target, y_pred))