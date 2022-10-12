import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from P3_Dataset import P3Dataset
from torchvision import transforms
from P3_Model import DANN, FeatureExtractor
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source', help='source_domain', type=str)
    parser.add_argument('-t', '--target', help='target_domain', type=str)
    args = parser.parse_args()

    source = args.source
    target = args.target


    # hyperparam
    torch.random.manual_seed(4)
    batch_size = 15
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # dataset
    sroot = (f"../hw2-yitinghung/hw2_data/digits/{source}/test")
    s_csv = (f"../hw2-yitinghung/hw2_data/digits/{source}/test.csv")

    troot = (f"../hw2-yitinghung/hw2_data/digits/{target}/test")
    t_csv = (f"../hw2-yitinghung/hw2_data/digits/{target}/test.csv")

    transform = transforms.Compose([transforms.ToTensor()])

    s_dataset = P3Dataset(root=sroot, csv_pth=s_csv, transform=transform)
    s_dataloader = torch.utils.data.DataLoader(s_dataset, batch_size=batch_size,shuffle=True)

    t_dataset = P3Dataset(root=troot,csv_pth=t_csv,transform=transform)
    t_dataloader = torch.utils.data.DataLoader(t_dataset, batch_size=batch_size,shuffle=True)


    # load model
    checkpoint = f"p3/{target}.pth"
    checkpoint = torch.load(checkpoint)

    model = DANN(device).to(device)
    model.load_state_dict(checkpoint)


    # calculate accuracy
    test_correct = 0
    df_all = pd.DataFrame()
    model.eval()
    for i, data in enumerate(t_dataloader):
        with torch.no_grad():
            file_names , test_img, test_label = data
            test_img = test_img.to(device)
            test_label = test_label.to(device)
            
            class_output, domain_output = model(test_img)

            test_prob, test_pred = torch.max(class_output.data, 1)
            test_correct += (test_pred == test_label.long()).sum().item()

            df = pd.DataFrame({"image_name":list(file_names), 
                                "label"     :test_pred.detach().cpu().tolist()})
            df_all = df_all.append(df, ignore_index = True)

    df_all.to_csv('test.csv', index=0)
            
    test_acc = float(test_correct) / len(t_dataset)
    item_pr = 'test_acc:{:.4f}'.format(test_acc)
    print(item_pr)
