import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
from model import Convnet

filenameToPILImage = lambda x: Image.open(x)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            Image.open,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def euclidean_metric(query, mean):
    n_quary = query.shape[0]
    n_way = mean.shape[0]
    query = query.unsqueeze(1).expand(n_quary, n_way, -1)
    mean = mean.unsqueeze(0).expand(n_quary, n_way, -1)
    logits = -((query - mean)**2).sum(dim=2)
    return logits

def predict(args, model, device, data_loader):
    model = model.to(device)
    
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            #print(i)
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.tensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]])
            query_label = query_label.long()
            
            # TODO: extract the feature of support and query data
            support = model(support_input)
            query = model(query_input)
            
            # TODO: calculate the prototype for each class according to its support data
            prototype = support.reshape(args.N_shot, args.N_way, -1).mean(dim=0)
            
            # TODO: classify the query data depending on the its distense with each prototype
            logits = euclidean_metric(query, prototype)
            predict = torch.argmax(logits, dim=1)
            prediction_results.append(predict.tolist())
    
    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', default = 'p1/model.pth', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    DEVICE = torch.device('cpu')
    #DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Convnet()
    checkpoint = torch.load(args.load , map_location = 'cpu')
    model.load_state_dict(checkpoint)

    prediction_results = predict(args, model, DEVICE, test_loader)
    
    # # TODO: output your prediction to csv
    df = pd.DataFrame(prediction_results, columns =[f'query{i}' for i in range(75)]).rename_axis('episode_id')
    output_pth = os.path.join(args.output_csv)
    df.to_csv(output_pth, index=1)
    
    
    
    
    
    
    
    
    