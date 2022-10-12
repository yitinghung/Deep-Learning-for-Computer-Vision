import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import testDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import pandas as pd
import argparse
import random

def seed(SEED=123):
    # fix random seeds for reproducibility
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

def test(model, test_loader, output_pth, label_dic):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    model.eval()
    df_total = pd.DataFrame()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            img, fn, id = data
            img = img.to(device)
            logits = model(img)

            test_pred = logits.argmax(dim=-1)
            test_pred = test_pred.cpu().tolist()
            pred_name = [list(label_dic.keys())[list(label_dic.values()).index(test_pred[i])] for i in range(len(test_pred))]

            df = pd.DataFrame({"id":id.tolist(), "filename": list(fn), 
                                "label": pred_name})
            df_total = df_total.append(df, ignore_index = True)
        df_total.to_csv(output_pth, index=0)
        
    
      
if __name__ == '__main__':
    seed(123)
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('-c', '--csv_file', type=str, help='path to test image csv file', required=True)
    parser.add_argument('-i', '--img_path', type=str, help='path to test image folder', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='path to the output csv file', required=True)
    args = parser.parse_args()

    cvs_pth = args.csv_file
    test_root = args.img_path
    output_pth = args.output_path
    #ckpt_pth = '/home/yiting/Documents/DLCV/hw4/p2/log/C_finetune_ep340_0.4729.pth'
    ckpt_pth = './p2/model.pth'

    batchSize = 64
    label_dic = {'Fork': 0, 'Radio': 1, 'Glasses': 2, 'Webcam': 3, 'Speaker': 4, 'Keyboard': 5, 'Sneakers': 6, 'Bucket': 7, 'Alarm_Clock': 8, 'Exit_Sign': 9, 'Calculator': 10, 'Folder': 11, 'Lamp_Shade': 12, 'Refrigerator': 13, 'Pen': 14, 'Soda': 15, 'TV': 16, 'Candles': 17, 'Chair': 18, 'Computer': 19, 'Kettle': 20, 'Monitor': 21, 'Marker': 22, 'Scissors': 23, 'Couch': 24, 'Trash_Can': 25, 'Ruler': 26, 'Telephone': 27, 'Hammer': 28, 'Helmet': 29, 'ToothBrush': 30, 'Fan': 31, 'Spoon': 32, 'Calendar': 33, 'Oven': 34, 'Eraser': 35, 'Postit_Notes': 36, 'Mop': 37, 'Table': 38, 'Laptop': 39, 'Pan': 40, 'Bike': 41, 'Clipboards': 42, 'Shelf': 43, 'Paper_Clip': 44, 'File_Cabinet': 45, 'Push_Pin': 46, 'Mug': 47, 'Bottle': 48, 'Knives': 49, 'Curtains': 50, 'Printer': 51, 'Drill': 52, 'Toys': 53, 'Mouse': 54, 'Flowers': 55, 'Desk_Lamp': 56, 'Pencil': 57, 'Sink': 58, 'Batteries': 59, 'Bed': 60, 'Screwdriver': 61, 'Backpack': 62, 'Flipflops': 63, 'Notebook': 64}


    model = torchvision.models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
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


    checkpoint = torch.load(ckpt_pth)
    model.load_state_dict(checkpoint['state_dict'])


    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    test_set = testDataset(test_root, cvs_pth, transform)
    test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=False)

    test(model, test_loader, output_pth, label_dic)
