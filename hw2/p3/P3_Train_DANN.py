import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from P3_Dataset import P3Dataset
from torchvision import transforms
from P3_Model import DANN
import random


def train(model, optimizer, sloader, tloader, model_path, device):
    src_len = len(sloader)
    tar_len = tloader
    len_dataloader = min(src_len, tar_len)
    criterion = torch.nn.CrossEntropyLoss()
    #criterion = nn.NLLLoss()
    for epoch in range(num_epoch):
        model.train()
        i = 1
        for (simg, timg) in zip(enumerate(sloader), enumerate(tloader)):
            index , (_, simg, slabel) = simg
            index , (_, timg, tlabel) = simg
            simg, slabel, timg = simg.to(device), slabel.to(device), timg.to(device)

            p = float(i + epoch * len_dataloader) / num_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            pred_label, err_s_domain = model(simg, alpha)
            err_s_label = criterion(pred_label, slabel)
            
            _, err_t_domain = model(timg, alpha, source=False)
            err_domain = err_t_domain + err_s_domain
            err = err_s_label + 0.5 * err_domain
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            i += 1
    

        # test
        s_acc = test_acc(model, sloader, device)
        t_acc = test_acc(model, tloader, device)
        torch.save(model.state_dict(), '{}/{}_{:.4f}.pth'.format(model_path, epoch, t_acc))
        
        print(f'Epoch: [{epoch}/{num_epoch}], classify_loss: {err_s_label.item():.4f}, domain_loss_s: {err_s_domain.item():.4f}, domain_loss_t: {err_t_domain.item():.4f}, domain_loss: {err_domain:.4f},total_loss: {err.item():.4f}')
        print(f'Source accuracy: {s_acc:.4f}, Target accuracy: {t_acc:.4f}\n')


def test_acc(model, dataloader, device):
    alpha = 0
    model.eval()
    correct = 0
    with torch.no_grad():
        for _, (_ ,img, label) in enumerate(dataloader):
            img, label = img.to(device), label.to(device)
            pred_label, _ = model(img, alpha)
            _, pred = torch.max(pred_label.data, 1)
            correct += (pred == label).sum().item()
    correct = float(correct)
    accuracy = correct / len(dataloader.dataset)
    return accuracy       


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
    # hyperparam
    same_seeds(4)
    lr = 1e-3
    batch_size = 512
    num_epoch = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model_path = './checkpoints/US'
    if not os.path.exists(model_path):
    	os.makedirs(model_path)
    model = DANN(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.999))
    
    
    #% load data
    sroot = "../hw2-yitinghung/hw2_data/digits/usps/train"
    s_csv_pth = "../hw2-yitinghung/hw2_data/digits/usps/train.csv"

    sset = P3Dataset(root=sroot, csv_pth=s_csv_pth, transform=transforms.Compose([transforms.ToTensor()]))
    sloader = DataLoader(sset, batch_size=batch_size,shuffle=True)
    
    troot = "../hw2-yitinghung/hw2_data/digits/svhn/train"
    t_csv_pth = "../hw2-yitinghung/hw2_data/digits/svhn/train.csv"
    
    tset = P3Dataset(root=troot, csv_pth=t_csv_pth, transform=transforms.Compose([transforms.ToTensor()]))
    tloader = DataLoader(tset, batch_size=batch_size,shuffle=True)
    

    train(model, optimizer, sloader, tloader, model_path, device)
    
