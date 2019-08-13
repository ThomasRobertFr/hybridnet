import torchvision.transforms as transforms

from . import data
from .utils import export


@export
def imagenet():
    channel_stats = dict(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    train_transformation = data.TransformTwice(transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data/processed/ilsvrc2012/',
        'num_classes': 1000
    }


@export
def cifar10():
    channel_stats = dict(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470,  0.2435,  0.2616])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data/processed/cifar10/images/',
        'num_classes': 10
    }

@export
def svhn():
    channel_stats = dict(mean=[109.86000817/255, 109.67725731/255, 113.76569232/255],
                         std=[50.11031433/255, 50.56369511/255, 50.84907838/255])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data/processed/svhn/images/',
        'num_classes': 10
    }

@export
def stl10():
    channel_stats = dict(mean=[0.44058639, 0.42731114, 0.38579501],
                         std=[0.26865456, 0.26125219, 0.26849921])
    train_transformation = data.TransformTwice(transforms.Compose([
        data.RandomTranslateWithReflect(8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ]))
    eval_transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**channel_stats)
    ])

    return {
        'train_transformation': train_transformation,
        'eval_transformation': eval_transformation,
        'datadir': 'data/processed/stl10/images/',
        'num_classes': 10
    }
