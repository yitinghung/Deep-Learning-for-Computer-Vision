from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

class myDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.filenames = [file for file in os.listdir(root)]
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(self.filenames[index].split('_')[0])
        return img, label, self.filenames[index]
    
    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    root = '/Users/francine/Documents/DLCV2021/hw3/hw3_data/p1_data/train'
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    dataset = myDataset(root, transform)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    dataiter = iter(dataloader)
    img, label, fn = dataiter.next()
    plt.imshow(np.transpose(utils.make_grid(img), (1, 2, 0)))
    plt.show()
