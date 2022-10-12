import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import myDataset
from torch import optim
from PIL import Image
import numpy as np
import random
import os

def seed(SEED=123):
    # fix random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

def save_model(path, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def fix_backbone(m):
    if isinstance(m, nn.Conv2d):
        m.requires_grad_(False)

def train(model, optimizer, train_loader, val_loader, num_epoch, lr, ckpt_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    best_acc = 0.0

    model.train()
    for epoch in range(num_epoch):
        #---------Training---------
        train_acc = 0.0
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            img, label, fn, id = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_acc += torch.sum(logits.argmax(dim=-1) == label).float()   
            train_loss += loss.item()

        train_acc = train_acc / len(train_loader.dataset)
        train_loss = train_loss / len(train_loader.dataset)
        train_acc_list.append(train_acc)
        train_loss_list.append(train_loss)

        #---------Validation--------
        model.eval()
        val_acc = 0.0
        val_loss = 0.0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                img, label, fn, id = data
                img, label = img.to(device), label.to(device)
                logits = model(img)
                loss = criterion(logits, label)
                val_acc += torch.sum(logits.argmax(dim=-1) == label).float()
                val_loss += loss.item()

            val_acc = val_acc / len(val_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)
            if val_acc > best_acc:
                best_acc = val_acc
                save_model(path=ckpt_pth, model=model, optimizer=optimizer)
                print(f'Saving model with acc {best_acc:.3f}')
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)

        print(f'[Epoch: {epoch+1:03d}/{num_epoch:03d}]  Train Acc: {train_acc:.4f}  Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}  Loss: {val_loss:.4f}')
      

if __name__ == '__main__':
    batchSize = 128
    lr = 0.0001
    seed(123)
    num_epoch = 100
    ckpt_pth = 'log/C_finetune_ep360.pth'    

    train_root = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/train'
    train_csv = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/train.csv'
    val_root = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/val'
    val_csv = '/home/yiting/Documents/DLCV/hw4/hw4_data/office/val.csv'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    train_set = myDataset(train_root, train_csv, transform)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)

    val_set = myDataset(val_root, val_csv, transform)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    model = torchvision.models.resnet50(pretrained=False)
    #pretrain_ckpt_pth = '/home/yiting/Documents/DLCV/hw4/hw4_data/pretrain_model_SL.pt'
    pretrain_ckpt_pth = '/home/yiting/Documents/DLCV/hw4/byol-pytorch/log/C_pretrain_ep360'
    checkpoint = torch.load(pretrain_ckpt_pth)
    
    model.load_state_dict(checkpoint['state_dict'])

    num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 65)
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 65),
    )

    ## fix backbone
    # print(model.conv1.weight.requires_grad)
    # model.apply(fix_backbone)
    # print(model.conv1.weight.requires_grad)


    FOUND_LR = 1e-3
    params = [
            {'params': model.conv1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.bn1.parameters(), 'lr': FOUND_LR / 10},
            {'params': model.layer1.parameters(), 'lr': FOUND_LR / 8},
            {'params': model.layer2.parameters(), 'lr': FOUND_LR / 6},
            {'params': model.layer3.parameters(), 'lr': FOUND_LR / 4},
            {'params': model.layer4.parameters(), 'lr': FOUND_LR / 2},
            {'params': model.fc.parameters()}
            ]
    #optimizer = optim.SGD(params, lr=FOUND_LR, momentum=0.9)
    optimizer = optim.Adam(params, lr = FOUND_LR)
    #print(model)

    train(model, optimizer, train_loader, val_loader, num_epoch, lr, ckpt_pth)

