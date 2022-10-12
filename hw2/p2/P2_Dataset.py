from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd

class P2Dataset(Dataset):
    def __init__(self, root, csv_pth, transform=None):
        self.root = root
        self.transform = transform
        #self.filenames = [os.path.join(self.root, file) for file in os.listdir(self.root)]
        self.filenames = [file for file in os.listdir(self.root)]
        #self.df_label = pd.read_csv(os.path.join(self.root, 'train.csv'))
        self.len = len(self.filenames)
        self.csv_pth = csv_pth

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.filenames[index]))
        if self.transform is not None:
            image = self.transform(image)

        df = pd.read_csv(self.csv_pth)
        label_index = df[df.image_name==self.filenames[index]].index.item()
        label = df['label'][label_index]

        #print(self.filenames[index], label)
        return image, label


    def __len__(self):
        return self.len

if __name__ == '__main__':
    root = '/home/yiting/Documents/DLCV/hw2/hw2-yitinghung/hw2_data/digits/mnistm/train'
    csv_pth = '/home/yiting/Documents/DLCV/hw2/hw2-yitinghung/hw2_data/digits/mnistm/train.csv'
    #transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = P2Dataset(root, csv_pth, transform = transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    print(labels)
    plt.imshow(np.transpose(utils.make_grid(images), (1, 2, 0)))
    plt.show()

