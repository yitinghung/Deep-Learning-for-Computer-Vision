from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np
import pandas as pd

label_dic = {'Fork': 0, 'Radio': 1, 'Glasses': 2, 'Webcam': 3, 'Speaker': 4, 'Keyboard': 5, 'Sneakers': 6, 'Bucket': 7, 'Alarm_Clock': 8, 'Exit_Sign': 9, 'Calculator': 10, 'Folder': 11, 'Lamp_Shade': 12, 'Refrigerator': 13, 'Pen': 14, 'Soda': 15, 'TV': 16, 'Candles': 17, 'Chair': 18, 'Computer': 19, 'Kettle': 20, 'Monitor': 21, 'Marker': 22, 'Scissors': 23, 'Couch': 24, 'Trash_Can': 25, 'Ruler': 26, 'Telephone': 27, 'Hammer': 28, 'Helmet': 29, 'ToothBrush': 30, 'Fan': 31, 'Spoon': 32, 'Calendar': 33, 'Oven': 34, 'Eraser': 35, 'Postit_Notes': 36, 'Mop': 37, 'Table': 38, 'Laptop': 39, 'Pan': 40, 'Bike': 41, 'Clipboards': 42, 'Shelf': 43, 'Paper_Clip': 44, 'File_Cabinet': 45, 'Push_Pin': 46, 'Mug': 47, 'Bottle': 48, 'Knives': 49, 'Curtains': 50, 'Printer': 51, 'Drill': 52, 'Toys': 53, 'Mouse': 54, 'Flowers': 55, 'Desk_Lamp': 56, 'Pencil': 57, 'Sink': 58, 'Batteries': 59, 'Bed': 60, 'Screwdriver': 61, 'Backpack': 62, 'Flipflops': 63, 'Notebook': 64}

class myDataset(Dataset):
    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform
        self.data_df = pd.read_csv(csv_path)
        self.ids = self.data_df["id"].tolist()
        self.filenames = self.data_df["filename"].tolist()
        self.labels = self.data_df["label"].tolist()
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label_name = self.labels[index]
        label = label_dic[label_name]
        return img, label, self.filenames[index], self.ids[index]
    
    def __len__(self):
        return len(self.filenames)

class testDataset(Dataset):
    def __init__(self, root, csv_path, transform=None):
        self.root = root
        self.transform = transform
        self.data_df = pd.read_csv(csv_path)
        self.ids = self.data_df["id"].tolist()
        self.filenames = self.data_df["filename"].tolist()
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.filenames[index], self.ids[index]
    
    def __len__(self):
        return len(self.filenames)

class BYOLDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.filenames = [file for file in os.listdir(root)]
        self.transform = transform
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.filenames[index]))
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    root = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/train'
    cvs_pth = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/train.csv'
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    dataset = myDataset(root, cvs_pth, transform)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    dataiter = iter(dataloader)
    img, label, fn, id = dataiter.next()
    print(label, fn)
    plt.imshow(np.transpose(utils.make_grid(img), (1, 2, 0)))
    plt.show()
