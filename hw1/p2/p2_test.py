from FCN16 import FCN16
from func import val_tfm

import os
import torch
import numpy as np
from myDataset import myDataset
from viz_mask import cls_color
import scipy.misc
import argparse 
import imageio

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_dir', help='img_dir', type=str)
    parser.add_argument('-o', '--output_dir', help='output_dir', type=str)
    args = parser.parse_args()
    #/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p2_data/validation

    print(args.img_dir)
    print(args.output_dir)
    print(16)
    print(torch.__version__)

    #%% Path
    model_pth = 'p2_model.ckpt'
    path_out = args.output_dir
    root = args.img_dir

    #%% Other params
    prediction = True
    feature_extract = True
    num_classes = 7
    batch_size = 4
    device = get_device()
    
    #%% Test Dataset
    val_set = myDataset(root=root, transform=val_tfm, prediction=prediction)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
      

    #%% Model
    model = FCN16(num_classes=num_classes, feature_extract=feature_extract)
    #% Load checkpoint
    checkpoint = torch.load(model_pth, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    model = model.to(device)
    
                
    #%% save
    cmap = cls_color
    with torch.no_grad():
        for i, (imgs, _, file_name) in enumerate(val_loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            pred = torch.argmax(outputs, dim=1).unsqueeze(1)
            pred_cpu = pred.squeeze(1).detach().cpu().numpy()
            
            
            # transfer index 2 color & save
            mask = pred.transpose(1, 2).transpose(2, 3)
            for i in range(mask.size(0)):
                mask_save = np.empty([512, 512, 3])
                npmask = pred_cpu[i,:,:]
                indexs = np.unique(npmask)
                for index in indexs:
                    mask_save[npmask==index] = cmap[index]
                
                fn = file_name[i].replace('.jpg', '.png')
                path_mask = os.path.join(path_out, fn)
                #path_mask = os.path.join(path_out, file_name[i])
                imageio.imwrite(path_mask, np.uint8(mask_save))

    
    