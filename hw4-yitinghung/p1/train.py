import numpy as np
import os
import argparse
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_1 import MiniDataset
from sampler import CategoriesSampler
from model import Convnet

def euclidean_metric(query, mean):
    n_quary = query.shape[0]
    n_way = mean.shape[0]
    query = query.unsqueeze(1).expand(n_quary, n_way, -1)
    mean = mean.unsqueeze(0).expand(n_quary, n_way, -1)
    logits = -((query - mean)**2).sum(dim=2)
    
    return logits

def cosine_similarity(query, mean):                                                            
    n_quary = query.shape[0]
    n_way = mean.shape[0]
    query = query.unsqueeze(1).expand(n_quary, n_way, -1)
    mean = mean.unsqueeze(0).expand(n_quary, n_way, -1)
    dot = (query*mean).sum(dim=2)
    norm_query = (query*query).sum(dim=2)**0.5
    norm_mean = (mean*mean).sum(dim=2)**0.5
    
    return dot/(norm_mean * norm_query)


def Manhattan_Distance(query, mean):
    n_quary = query.shape[0]
    n_way = mean.shape[0]
    query = aquery.unsqueeze(1).expand(n_quary, n_way, -1)
    mean = mean.unsqueeze(0).expand(n_quary, n_way, -1)
    logits = -(abs(query - mean)).sum(dim=2)
    
    return logits

class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def save_model(name):
    torch.save(model.state_dict(), osp.join(output_path, name + '.pth'))
    


if __name__ == '__main__':
    
    # Decide which device we want to run on
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    episode = 600
    lr = 0.001
    max_epoch = 200
    save_interval = 20

    train_classes = 64
    test_classes = 16
    shot = 1
    query = 15
    train_way = 10     # train 10-way 1-shot
    test_way= 5        # test 5-way 1-shot

    #output_path = f'log/cosine_similarity{result}'
    #output_path = f'log/euclidean_metric{result}'
    #output_path = f'log/Manhattan_Distance{result}'
    output_path = f'log/train_test_5way/10shot/euclidean'
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    train_set = MiniDataset(csv_path='../hw4_data/mini/train.csv', data_dir='../hw4_data/mini/train')
    train_sampler = CategoriesSampler(n_batch=episode, total_cls=train_classes , n_way=train_way, n_img=shot + query, n_shot=shot)
    train_loader = DataLoader(dataset=train_set, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=False)
    
    val_set = MiniDataset(csv_path='../hw4_data/mini/val.csv',data_dir='../hw4_data/mini/val')
    val_sampler =CategoriesSampler(n_batch=episode, total_cls=test_classes , n_way=test_way, n_img=shot + query, n_shot=shot)
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=False)
    
    model = Convnet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    max_acc = 0.0
    for epoch in range(1, max_epoch + 1):
        lr_scheduler.step()
        #------------Train--------------
        model.train()

        train_loss = Averager()
        train_acc = Averager()

        for i, (data, label) in enumerate(train_loader, 1):
            data = data.to(device)
            
            p = shot * train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, train_way, -1).mean(dim=0)

            label = torch.arange(train_way).repeat(query)
            label = label.long().to(device)

            #logits = cosine_similarity(model(data_query), proto)
            logits = euclidean_metric(model(data_query), proto)
            #logits = Manhattan_Distance(model(data_query), proto)
        
            loss = F.cross_entropy(logits, label)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()


            train_loss.add(loss.item())
            train_acc.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('     {} optim: {}'.format(i, optimizer.param_groups[0]['lr']))
            #lr_scheduler.step()
            #print('     {} scheduler: {}'.format(i, lr_scheduler.get_lr()[0]))

        train_loss = train_loss.item()
        train_acc = train_acc.item()
    
        print('[ Train ] [ epoch {epoch}/{max_epoch} ] loss: {train_loss:.4f} acc: {train_acc:.4f}'
        

        #------------Evaluation--------------
        model.eval()

        val_loss = Averager()
        val_acc = Averager()

        for i, (data, _) in enumerate(val_loader, 1):
            data = data.to(device)

            p = shot * test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(shot, test_way, -1).mean(dim=0)

            label = torch.arange(test_way).repeat(query)
            label = label.long().to(device)

            #logits = cosine_similarity(model(data_query), proto)
            logits = euclidean_metric(model(data_query), proto)
            #logits = Manhattan_Distance(model(data_query), proto)

            loss = F.cross_entropy(logits, label)
            pred = torch.argmax(logits, dim=1)
            acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

            val_loss.add(loss.item())
            val_acc.add(acc)
            

        if val_acc > max_acc:
            max_acc = val_acc
            save_model('max_acc')

        val_loss = val_loss.item()
        val_acc = val_acc.item()
        print(f'[ Val ] [epoch {epoch}/{max_epoch}] loss: {val_loss:.4f} acc: {val_acc:.4f}')

        save_model('last_ep')

        if epoch % save_interval == 0:
            save_model(f'ep{epoch}'))

    