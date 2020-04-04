import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
from data_loading import ZipLoader, UniformLoader
from torch.utils.data import DataLoader
from PIL import Image

try:
    device = torch.device("cuda")
    is_cuda = True
except:
    device = torch.device("cpu")
    is_cuda = False

tr = transforms.Compose([transforms.Resize((16, 16)),
                         transforms.ToTensor()])

mnist_small_train = dset.MNIST("data/mnist", download=True, transform=tr, train=True)
mnist_small_test = dset.MNIST("data/mnist", download=True, transform=tr, train=False)

mnist_train = dset.MNIST("data/mnist", download=True, transform=transforms.ToTensor(), train=True)
mnist_test = dset.MNIST("data/mnist", download=True, transform=transforms.ToTensor(), train=False)

usps_train = dset.USPS("data/usps", download=True, transform=transforms.ToTensor(), train=True)
usps_test = dset.USPS("data/usps", download=True, transform=transforms.ToTensor(), train=False)

# Example of pairs sampling
zl = ZipLoader([mnist_small_train, usps_train], batch_size=5, n_batches=2, pin_memory=is_cuda)

for X, Y in zl:
    print(X.shape, Y.shape)

# Example of sampling without splits
dl = UniformLoader(mnist_train, batch_size=5, n_batches=2, pin_memory=is_cuda)

for X, y in dl:
    print(X.shape, y.shape)