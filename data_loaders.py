import torch
from torchvision import datasets, transforms

def get_data_loader(dataset, batch_size, train=True, shuffle=True, drop_last=True):
    # Note that we do not normalize in the data loader, because we may use adv. examples
    # during training or testing.
    if dataset not in ('mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'):
        raise NotImplementedError('Dataset not supported.')
    if dataset == 'mnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.MNIST('./data', train=train, transform=tr, download=True)
    if dataset == 'fmnist':
        tr = transforms.Compose([
            transforms.ToTensor(),
        ])
        d = datasets.FashionMNIST('./data', train=train, transform=tr, download=True)
    elif dataset == 'cifar10':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        d = datasets.CIFAR10('./data', train=train, transform=tr, download=True)
    elif dataset == 'cifar100':
        if train:
            tr = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        d = datasets.CIFAR100('./data', train=train, transform=tr, download=True)
    elif dataset == 'svhn':
        if train:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        else:
            tr = transforms.Compose([
                transforms.ToTensor(),
            ])
        split = 'train' if train else 'test'
        d = datasets.SVHN('./data', split=split, transform=tr, download=True)
    data_loader = torch.utils.data.DataLoader(d, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    if train:
        return data_loader, None
    else:
        subset_dataset = torch.utils.data.Subset(d, range(1000))
        subset_loader = torch.utils.data.DataLoader(dataset=subset_dataset, batch_size=batch_size, shuffle=False)
        return data_loader, subset_loader
