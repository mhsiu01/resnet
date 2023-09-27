import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
# ffcv: Writing dataset
import ffcv
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
# ffcv: Loading dataset
from ffcv.fields.decoders import SimpleRGBImageDecoder, IntDecoder
from ffcv.transforms import ToTensor, ToDevice, NormalizeImage, ToTorchImage, Convert, Squeeze, RandomHorizontalFlip, RandomTranslate
from ffcv.loader import Loader, OrderOption

def _load_datasets(totensor, lengths, num_classes, seed=666):
    # load torchvision dataset: CHW in [0,1] scale
    T = [transforms.ToTensor()] if totensor else []
    T = transforms.Compose(T)
    # Split into training and validation
    if num_classes==100:
        train_val = torchvision.datasets.CIFAR100(
            './data/CIFAR100', download=True, train=True, transform=T)
        train, val = torch.utils.data.random_split(
            train_val, lengths=lengths, generator=torch.Generator().manual_seed(seed))
        test = torchvision.datasets.CIFAR100(
            './data/CIFAR100', download=True, train=False, transform=T)
        datasets = {
            'train':train,
            'val':val,
            'test':test
        }
        return datasets
    elif num_classes==10:
        train_val = torchvision.datasets.CIFAR10(
            './data/CIFAR10', download=True, train=True, transform=T)
        train, val = torch.utils.data.random_split(
            train_val, lengths=lengths, generator=torch.Generator().manual_seed(seed))
        test = torchvision.datasets.CIFAR10(
            './data/CIFAR10', download=True, train=False, transform=T)
        datasets = {
            'train':train,
            'val':val,
            'test':test
        }
        return datasets
    
def write_datasets(num_classes, lengths=[40000,10000], write=True):
    if write:
        # write to .betons
        datasets = _load_datasets(totensor=False, lengths=lengths, num_classes=num_classes)
        for split,dataset in datasets.items():
            if len(dataset)==0:
                continue
            writer = DatasetWriter(
                f'./data/ffcv/cifar{num_classes}_{split}.beton',
                {'image': RGBImageField(),
                 'label': IntField()},
                num_workers=4) 
            writer.from_indexed_dataset(dataset)
            print(f"Cifar{num_classes} split \'{split}\' containing {len(dataset)} examples written to .beton.")
    else:
        print("Using existing .beton files.")

    # get mean and std
    datasets = _load_datasets(totensor=True, lengths=lengths, num_classes=num_classes)
    loader = DataLoader(datasets['train'], batch_size=128, shuffle=False, drop_last=False)
    arr = torch.cat([x for x,_ in loader], dim=0)
    MEAN,STD = arr.mean(dim=(0,2,3)).numpy(),arr.std(dim=(0,2,3)).numpy()
    return MEAN,STD

def get_loader(MEAN, STD, split, num_classes, BS=128, SCALER=4, USE_FLOAT16=True, pad=4, NW=4, visualize=False, DEVICE=torch.device('cuda:0')):
    augmentations = [RandomHorizontalFlip(),RandomTranslate(padding=pad)] if split=='train' else []
    if visualize:
        augmentations = []
        order = OrderOption.SEQUENTIAL
    else:
        order = OrderOption.RANDOM
        
    PIPELINES = {
        'image':
            [SimpleRGBImageDecoder(),
            NormalizeImage(MEAN,STD,type=np.float16)] +
            augmentations +
            [ToTensor(),
            ToTorchImage(channels_last=True),
            ToDevice(DEVICE),
            Convert(torch.float16 if USE_FLOAT16 else torch.float32)],
        'label':
            [IntDecoder(),
            ToTensor(),
            ToDevice(DEVICE),
            Squeeze()]
    }
    loader = Loader(
        f'./data/ffcv/cifar{num_classes}_{split}.beton',
        batch_size=BS*SCALER,
        num_workers=NW,
        order=order,
        pipelines=PIPELINES,
        drop_last=False)
    return loader

# Example usage:
#     MEAN,STD = write_datasets(lengths=[40000,10000])
#     loader = get_loader(MEAN, STD, split='train')