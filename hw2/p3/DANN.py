import torch
import argparse
from torch.utils.data import DataLoader

from P3_Dataset import PredDataset
from torchvision import transforms
from P3_Model import DANN
import pandas as pd



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='input_dir', type=str)
    parser.add_argument('-t', '--target', help='target', type=str)
    parser.add_argument('-o', '--output_dir', help='output_dir', type=str)
    args = parser.parse_args()
    
    torch.random.manual_seed(4)

    print(args.input_dir)
    print(args.target)
    print(args.output_dir)

    input_dir = args.input_dir
    target_domain = args.target
    output_dir = args.output_dir
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 15
    transform = transforms.Compose([transforms.ToTensor()])
    
    # dataset
    dataset = PredDataset(root=input_dir,transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=None)
    
    # load model
    model = DANN(device).to(device)
    checkpoint = f'p3/{target_domain}.pth'
    checkpoint = torch.load(checkpoint , map_location = 'cpu')
    model.load_state_dict(checkpoint)
    
    # predict
    df_all = pd.DataFrame()
    model.eval()
    for data in dataloader:
        with torch.no_grad():
            fn, img = data
            img = img.to(device)
            label, _ = model(img)          
            _, pred_label = torch.max(label.data, 1)
            
            df = pd.DataFrame({"image_name":list(fn), 
                                "label"    :pred_label.detach().cpu().tolist()})
            df_all = df_all.append(df, ignore_index = True)
    
    df_all.to_csv(output_dir, index=0)