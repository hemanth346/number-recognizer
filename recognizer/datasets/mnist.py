from __future__ import print_function 
import torch
from torchvision import datasets, transforms
import shutil

class MNIST:
    
    def update_transforms(transforms=[], train=True):
        pass

    @staticmethod
    def load_data(data_dir, train_transforms=None, test_transforms=None):
        """
        Download data to given directory in current machine using given transforms
        """
        if not train_transforms:
            train_transforms = transforms.Compose([
                                       transforms.RandomRotation((-7.0, 7.0)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])
        if not test_transforms:
            test_transforms = transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                       ])

        train = datasets.MNIST(data_dir, 
            train=True, 
            download=True, 
            transform=train_transforms
            )

        test = datasets.MNIST(data_dir, 
            train=False, 
            download=True, 
            transform=test_transforms
            )
        return train, test

    @staticmethod
    def get_stats():
        """
        Statistics around data
        """
        tmp_dir = './tmp_datadir'
        raw = datasets.MNIST(tmp_dir, 
            train=True, 
            download=True, 
            transform=transforms.Compose([
                transforms.ToTensor()
                ])
        )
        data = raw.train_data
        data = raw.transform(data.numpy())

        print('[ Train data stats ]')
        print(' - Numpy Shape:', raw.train_data.cpu().numpy().shape)
        print(' - Tensor Shape:', raw.train_data.size())
        print(' - min:', torch.min(data))
        print(' - max:', torch.max(data))
        print(' - mean:', torch.mean(data))
        print(' - std:', torch.std(data))
        print(' - var:', torch.var(data))
        shutil.rmtree(tmp_dir)

    @staticmethod
    def get_loaders(train_dataset, test_dataset, batch_size=64, cuda=False, seed=11, **kwargs):
        
        # For reproducibility
        if cuda:
            torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        dataloader_args = dict(shuffle=True, batch_size=batch_size*2, num_workers=4, pin_memory=True) if cuda else dict(shuffle=True, batch_size=batch_size)

        train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
        test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)

        return train_loader, test_loader
