import os

from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import CelebA, MNIST

class SmallCelebA(CelebA):
    def _check_integrity(self) -> bool:
        return True
    

def prepare_data_mnist(root: str = "./data", train_ratio: float=0.8):
    print("{0:=^80}".format("Preparing data MNIST"))
    os.makedirs(root, exist_ok=True)
    mnist_data = MNIST(root, train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), # to [0, 1]
        transforms.Resize((32, 32)),
        lambda x: (x - 0.5) * 2, # convert to [-1, 1]
    ]))
    len_data = len(mnist_data)
    print(len_data)
    train_idx = range(int(train_ratio * len_data))
    test_idx = range(int(train_ratio * len_data), len_data)
    train_set = Subset(mnist_data, train_idx)
    test_set = Subset(mnist_data, test_idx)
    return train_set, test_set


def prepare_data_celebA(root: str="./data", train_ratio: float=0.8, max_data=100000):
    print("{0:=^80}".format("Preparing data CelebA"))
    os.makedirs(root, exist_ok=True)
    data = SmallCelebA(root, download=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop((148, 148)),
        transforms.Resize((64, 64)),
        lambda x: (x - 0.5) * 2, # convert to [-1, 1]
    ]))
    len_data = min(len(data), max_data)
    print(len_data)
    train_idx = range(int(train_ratio * len_data))
    test_idx = range(int(train_ratio * len_data), len_data)
    train_set = Subset(data, train_idx)
    test_set = Subset(data, test_idx)
    return train_set, test_set

def prepare_data(dataset_type="mnist", train_ratio=0.8):
    if dataset_type == "mnist":
        train_set, test_set = prepare_data_mnist(train_ratio=train_ratio)
    else:
        train_set, test_set = prepare_data_celebA(train_ratio=train_ratio)
    return train_set, test_set