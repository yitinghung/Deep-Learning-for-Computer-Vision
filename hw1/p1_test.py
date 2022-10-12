import numpy as np
import pandas as pd 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
from sklearn.manifold import TSNE
import random
import os
import sys
import argparse


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



def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Construct Dataset
class myDataset(Dataset):
    def __init__(self, datadir, test, transform=None):
        self.datadir = datadir
        self.files = [(os.path.join(datadir, file), file) for file in os.listdir(datadir)]
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        filepath, filename = self.files[index]
        img = Image.open(filepath)
        if self.transform is not None:
            img = self.transform(img)

        # if self.test==True:
        #     return img, filename

        if self.test ==True:
            label = -100
            return img, label, filename
        else:
            label = int(filename.split('_')[0])
            return img, label

    def __len__(self):
        return len(self.files)


# Create Model
class VGG16_model(nn.Module):
    def __init__(self, numClasses=50):
        super(VGG16_model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            #nn.Dropout(),
        )
        #self.fc1 =  nn.Linear(4096, 4096)
        self.ReLU = nn.ReLU(inplace=True)
        self.Dropout = nn.Dropout()
        self.fc2 = nn.Linear(4096, numClasses)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        sec_last = self.classifier(x)
        #sec_last = self.fc1(x)
        x = self.ReLU(sec_last)
        x = self.Dropout(x)
        x = self.fc2(x)
        return x, sec_last


# Training
def Test(model, pred_save_pth):
    model = model.to(device)
    model.device = device

    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(),lr=lr, weight_decay=1e-3)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    #best_val = 0.0
    #--------Training--------
    #--------Validation--------
    model.eval()
    img_id = []
    predictions = []
    val_acc = []

    for i, data in enumerate(test_loader):
        imgs, labels, filename = data
        imgs, labels = imgs.to(device), labels.to(device)
        #imgs = imgs.to(device)
        with torch.no_grad():
            logits, _ = model(imgs)
        #print(logits[0].argmax(dim=-1).cpu().numpy().tolist())
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        #print(filename)
        #predictions.append(logits.argmax(dim=-1).cpu().numpy().tolist())
        #loss = criterion(logits, labels)
        
        acc = (logits.argmax(dim=-1) == labels).float().mean()  # acc應該也是一個tensor? append到val_acc時為何不用acc.item()?
        #val_loss.append(loss.item())
        img_id.extend(list(filename))
        #print(img_id)

        ##img_id.extend(labels.cpu().numpy().tolist())
        val_acc.append(acc)
    dic = {'image_id': img_id, 'label': predictions}
    df = pd.DataFrame(dic)
    df.to_csv(pred_save_pth, index=0)

    #val_loss = sum(val_loss) / len(val_loss)
    val_acc = sum(val_acc) / len(val_acc)
    #print(f'[ Validation ] acc = {val_acc:.5f}')

    torch.cuda.empty_cache() 
    #print(model.state_dict()['fc2.bias'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', help='img_dir', type=str)
    parser.add_argument('-o', '--output_dir', help='output_dir', type=str)
    args = parser.parse_args()

    print(args.img_dir)
    print(args.output_dir)
    print(torch.__version__)

    test_datadir = args.img_dir
    output_dir = args.output_dir

    # Config
    #train_datadir = '/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p1_data/train_50'
    #val_datadir = '/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p1_data/val_50'
    batch_size = 8
    lr = 0.0001
    #num_epochs = 1
    same_seeds(2021)
    device = get_device()


    train_tfm = transforms.Compose([
        #transforms.RandomApply(transform_set, p=0.5),
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_tfm = transforms.Compose([
        transforms.Resize(128),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Construct Dataloader
    #train_set = myDataset(datadir=train_datadir, test=False, transform=train_tfm)
    test_set = myDataset(datadir=test_datadir, test=True, transform=test_tfm)
    #train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    
    #vgg16 = models.vgg16(pretrained=True)
    model = VGG16_model(numClasses=50)
    #pretrained_dict = vgg16.state_dict()
    #model_dict = model.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    #model_dict.update(pretrained_dict)
    #model.load_state_dict(model_dict)
    #print(model.state_dict()['output.bias'])

    # from torchsummary import summary
    # summary(model, (3, 32, 32))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    checkpoint = torch.load('p1_model.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    Test(model, output_dir)




## Reference
# 使用pretrained vgg16: 
#   https://daniel820710.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92%E5%BE%9E%E9%9B%B6%E5%88%B0%E4%B8%80-day3-pytorch-%E4%BB%8B%E7%B4%B9%E8%88%87%E7%AF%84%E4%BE%8B-cosmetics-classification-6e826fbce59b
# load pretrained model的同時也修改其架構：
#   https://blog.csdn.net/whut_ldz/article/details/78845947
# 學長code
# ML hw3 sample code
# t-SNE:
#   https://mortis.tech/2019/11/program_note/664/
