from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import os
from PIL import Image 
import numpy as np
import random
import torch
import pandas as pd

def default_loader(path):
    return Image.open(path).convert('RGB')

class PredDataset(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = root
        self.fn = sorted(os.listdir(root))
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        file_name = self.fn[index]
        file_path = os.path.join(self.root, file_name)
        
        img = self.loader(file_path)
        if self.transform is not None:
            img = self.transform(img)
            
        return file_name, img

    def __len__(self):
        return len(self.fn)

class P3Dataset(Dataset):
    def __init__(self, root, csv_pth, transform=None):
        self.root = root
        self.transform = transform
        self.df = pd.read_csv(csv_pth)
        self.len = self.df.shape[0]

    def __getitem__(self, index):
        fn = self.df['image_name'][index]
        image = Image.open(os.path.join(self.root, fn))
        if self.transform is not None:
            image = self.transform(image)
        label = self.df['label'][index]
        return fn, image, label

    def __len__(self):
        return self.len

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    root = '/Users/francine/Documents/DLCV2021/hw2/hw2-yitinghung/hw2_data/digits/mnistm/try_train'
    csv_pth = '/Users/francine/Documents/DLCV2021/hw2/hw2-yitinghung/hw2_data/digits/mnistm/try_train.csv'
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = P3Dataset(root, csv_pth, transform = transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



    dataiter = iter(dataloader)
    fns, images, labels = dataiter.next()
    print(labels)
    plt.imshow(np.transpose(utils.make_grid(images), (1, 2, 0)))
    plt.show()

