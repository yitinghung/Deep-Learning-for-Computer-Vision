import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model
from P1_DataLoader import P1Dataset
from P1_Train import same_seeds
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch
from torchvision.utils import save_image
import os
import numpy as np
import random
import argparse

if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', help='output_path', type=str)
    args = parser.parse_args()

    print(f'output_path: {args.output_path}')
    output_path = args.output_path

    same_seeds(2021)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    nc = 3
    nz = 1024
    ngf = 64
    # samples 1000 noise vectors from Normal distribution and input them into your Generator.
    fixed_noise = torch.randn(1000, nz, 1, 1, device=device)

    model = torch.load('p1/p1_model.pth').to(device)
    print('model loaded!')

    model.eval()
    with torch.no_grad():
        fake = model(fixed_noise).detach().to(device)
        fake = (1+fake)/2
        for i in range(fake.shape[0]):
            save_image(fake[i], os.path.join(output_path, f'{i+1:04d}.png'))
        print(fake.shape)
        out = torchvision.utils.make_grid(fake[:32], padding=2, normalize=True)
        #save_image(out, 'output32.png')
        print('image saved!')

## 存1000張獨立照片，前32張放report

#finalG-0.94870.pth FID = 27.95498399607507 IS = 2.10238193851919
# 試到0.34696
