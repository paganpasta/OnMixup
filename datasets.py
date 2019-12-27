__author__ = "Aditya Singh"
__version__ = "0.1"

import torchvision
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_train_valid_loader(dataset='cifar100', batch_size=64, valid_size=0.1, num_workers=4, pin_memory=True, transform=None):

    if dataset.lower() == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    elif dataset.lower() == 'imagenet':
        dataset = torchvision.datasets.ImageNet(root='./data', train=True, download=True, transform=transform)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))

    train_num = int((1-valid_size)*len(dataset))
    val_num = len(dataset) - train_num
    print('Train/Val split {}/{} for {} total items'.format(train_num, val_num, len(dataset)))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_num, val_num])

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, valid_loader


def get_test_loader(dataset='cifar100', batch_size=64, num_workers=4, pin_memory=True):
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset.lower() == 'cifar100':
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    elif dataset.lower() == 'imagenet':
        test_dataset = torchvision.datasets.ImageNet(root='./data', train=False, download=True, transform=test_transform)
    else:
        raise AssertionError('Dataset {} is currently not supported'.format(dataset))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size
                                              , num_workers=num_workers, pin_memory=pin_memory)
    return test_loader


class TransformComposer:
    def __init__(self, transforms=None, inp_size=32, rotation=15):
        if transforms is None:
            transforms = ['crop', 'hflip', 'rotate']
        self.transforms = transforms
        self.inp_size = inp_size
        self.rotation = rotation

    def get_composite(self):
        composite_transforms = []
        for transform in self.transforms:
            if transform.lower() == 'crop':
                composite_transforms.append(transforms.RandomCrop(self.inp_size, padding=4))
            if transform.lower() == 'rotate':
                composite_transforms.append(transforms.RandomRotation(15))
            if transform.lower() == 'hflip':
                composite_transforms.append(transforms.RandomHorizontalFlip())
        composite_transforms.append(transforms.ToTensor())
        return transforms.Compose(composite_transforms)