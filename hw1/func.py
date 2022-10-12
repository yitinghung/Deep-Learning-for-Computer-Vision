import numpy as np
from torchvision import transforms


def write(type, info):
    print('-'*16)
    print("[", type, "]")
    print(info)
    
    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def train_tfm(image, mask):
    
    # Random horizontal flipping
    if np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    # Random vertical flipping
    if np.random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    
    # Normalize
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    
    return image, mask


def val_tfm(image, mask):
    
    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    
    # Normalize
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    
    return image, mask