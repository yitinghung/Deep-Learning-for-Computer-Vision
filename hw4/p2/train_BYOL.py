import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from dataset import BYOLDataset
from torch import optim
from PIL import Image
import numpy as np
import random
import os

from byol_pytorch import BYOL

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


def train(model, train_loader, num_epoch, lr, ckpt_pth, save_epoch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    learner = BYOL(
        model,
        image_size = 128,
        hidden_layer = 'avgpool'
    ).to(device)

    optimizer = torch.optim.Adam(learner.parameters(), lr=lr)

    model.train()
    for epoch in range(num_epoch):
        #---------Training---------
        train_loss = 0.0

        for i, data in enumerate(train_loader):
            img = data
            img = img.to(device)
            loss = learner(img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            train_loss += loss.item()

        train_loss = train_loss / len(train_loader.dataset)
        print(f'[Epoch:{epoch+1:04d}/{num_epoch:04d}]  Loss: {train_loss:.4f}')
        if epoch % save_epoch == 0:
            save_model(path=f'{ckpt_pth}_ep{epoch}', model=model, optimizer=optimizer)


if __name__ == '__main__':
    batchSize = 32
    lr = 3e-4
    num_epoch = 1000
    save_epoch = 20
    num_workers = 4
    ckpt_pth = 'log/C_pretrain'    

    train_root = '/home/yiting/Documents/DLCV/hw4/hw4_data/mini/train' 

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

    train_set = BYOLDataset(train_root, transform)
    train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True)

    model = torchvision.models.resnet50(pretrained=False)

    train(model, train_loader, num_epoch, lr, ckpt_pth, save_epoch)

