import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Dataset import myDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import timm
import argparse

def test(model, test_loader, output_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    test_acc = 0.0
    test_loss = 0.0
    df_total = pd.DataFrame()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, label, fn = data
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = criterion(logits, label)

            test_pred = logits.argmax(dim=-1)
            test_acc += torch.sum(test_pred == label).float()
            test_loss += loss.item()
            
            df = pd.DataFrame({"filename": list(fn), 
                                "label": test_pred.cpu().tolist()})
            df_total = df_total.append(df, ignore_index = True)
        df_total.to_csv(output_pth, index=0)
        
        test_acc = test_acc / len(test_loader.dataset)
        test_loss = test_loss / len(test_loader.dataset)
        
    print(f'Acc: {test_acc:.4f}')
      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('-i', '--img_path', type=str, help='path to test image folder', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='path to the output csv file', required=True)
    args = parser.parse_args()

    output_pth = args.output_path
    ckpt_pth = 'p1/model.pth'
    test_root = args.img_path
    batchSize = 32

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=37)
    checkpoint = torch.load(ckpt_pth)
    model.load_state_dict(checkpoint['state_dict'])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    test_set = myDataset(test_root, transform)
    test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=False)

    test(model, test_loader, output_pth)