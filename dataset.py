import os
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets
import pytorch_lightning as pl
import torch


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, target_transform=None):    
        super(ImageDataset, self).__init__()
        
        self.dataset = dataset
        
        self.transform = transform
        self.target_transform = target_transform
                    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        X, y = self.dataset[idx]
        
        if self.transform!=None:
            X = self.transform(image=X)["image"]
        if self.target_transform!=None:
            y = self.target_transform(image=y)["image"]
        
        X = torch.tensor(X).permute(2,0,1)
        y = torch.tensor(y)
        
        return X, y

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, cfg, transform=None, target_transform=None):
        super(ImageDataModule, self).__init__()
        
        self.cfg = cfg
        
        self.transform = transform
        self.target_transform = target_transform
    
    def prepare_data(self):
        self.train_dataset = datasets.CIFAR10(os.getcwd(), download=True, transform=np.array)

    def setup(self, stage=None):            
        n_train = int(len(dataset)*self.cfg.train_split+0.5)
        n_val = int(len(dataset)*(1-self.cfg.train_split)+0.5)
        
        imageDataset = ImageDataset(self.train_dataset, transform=self.transform, target_transform=self.target_transform)
        self.train_dataset, self.val_dataset = random_split(imageDataset, [n_train, n_val])
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.batch_size, num_workers=4, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.batch_size, num_workers=4, shuffle=False)
        return val_loader
