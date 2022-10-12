from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
import numpy as np

class P1Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = [os.path.join(self.root, file) for file in os.listdir(self.root)]
        self.len = len(self.filenames)

    def __getitem__(self, index):
        image = Image.open(self.filenames[index])
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        return self.len

if __name__ == '__main__':
    root = '/Users/francine/Documents/DLCV2021/hw2/hw2-yitinghung/hw2_data/face/train'
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = P1Dataset(root, transform = transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    dataiter = iter(dataloader)
    images = dataiter.next()
    plt.imshow(np.transpose(utils.make_grid(images), (1, 2, 0)))
    plt.show()

