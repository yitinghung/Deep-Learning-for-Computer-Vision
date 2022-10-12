import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Dataset import myDataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import random
import os
import timm
from sam import SAM

def save_model(path, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, path)

def train(model, train_loader, val_loader, num_epoch, lr, ckpt_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    #base_optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    base_optimizer = optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=lr, momentum=0.9)

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
            img, label, fn = data
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(img)
            # first forward-backward pass
            loss = criterion(logits, label)   # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            loss2 = criterion(model(img), label).backward()  # make sure to do a full forward pass & make sure not to reuse model outputs
            optimizer.second_step(zero_grad=True)

            train_acc += torch.sum(logits.argmax(dim=-1) == label).float()   #需要.float嗎？
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
                img, label, fn = data
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
    batchSize = 64
    lr = 0.0001
    num_epoch = 100
    ckpt_pth = 'p1/checkpoints/not_pretrained_base16_224.pth'    

    train_root = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/train'
    val_root = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/val'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    train_set = myDataset(train_root, transform)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)

    val_set = myDataset(val_root, transform)
    val_loader = DataLoader(val_set, batch_size=batchSize, shuffle=False)

    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=37)
    #model = timm.create_model('vit_tiny_patch16_224_in21k', pretrained=True, num_classes=37)
    print(torch.__version__)
    train(model, train_loader, val_loader, num_epoch, lr, ckpt_pth)

# Reference: https://medium.com/%E8%BB%9F%E9%AB%94%E4%B9%8B%E5%BF%83/sharpness-aware-minimization-sam-%E7%B0%A1%E5%96%AE%E6%9C%89%E6%95%88%E5%9C%B0%E8%BF%BD%E6%B1%82%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B-257613bb365