# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:55:36 2020

@author: opgg
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import tqdm
from mean_iou_evaluate import mean_iou_score

from P2Dataset import P2Dataset
from FCN32 import FCN32
from FCN16 import FCN16
from util import printw, transform_train, transform_val

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


def train_model(model, dataloaders, criterion, optimizer, num_epochs=1):
    since = time.time()
    
    training_loss = []
    validation_loss = []
    val_iou_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_iou = 0.0

    for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            mean_iou = 0.0
            
            with tqdm.tqdm(total=len(dataloaders[phase])) as pbar:
                pbar.set_description('epoch [{}/{}]: {} step'.format(epoch, num_epochs - 1, phase))
                # Iterate over data.
                for i, (inputs, labels, _) in enumerate(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        
                        outputs = model(inputs)
                        
                        loss = CrossEntropy2d(outputs, labels, criterion)
                        
                        preds = torch.argmax(outputs, dim=1).unsqueeze(1)
                        
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            training_loss.append(loss)
                        else:
                            validation_loss.append(loss)
                        
                        if (i+1) % 10 == 0:
                            pbar.set_postfix(Loss= loss.item())
                            pbar.update(10)
                    
                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    mean_iou += mean_iou_score(preds.squeeze(1).detach().cpu().numpy(), 
                                               labels.squeeze(1).detach().cpu().numpy())
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_iou = mean_iou / len(dataloaders[phase])
            print('')
            print('{} Loss: {:.4f} Iou: {:.4f}'.format(phase, epoch_loss, epoch_iou))
            
            # deep copy the model
            if phase == 'val' and epoch_iou > best_iou:
                best_iou = epoch_iou
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_iou_history.append(epoch_iou)
        
        print()
        
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Iou: {:4f}'.format(best_iou))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    history = {'training_loss':training_loss, 
               'validation_loss':validation_loss, 
               'val_iou_history':val_iou_history, }
    return model, history


def CrossEntropy2d(outputs, labels, criterion):
    outputs = outputs.transpose(1, 2).transpose(2, 3).contiguous()
    outputs = outputs.view(-1, 7)
    labels = labels.transpose(1, 2).transpose(2, 3).contiguous()
    labels = labels.view(-1)
    # print(outputs.size())
    # print(labels.size())
    loss = criterion(outputs, labels.long())
    return loss
    
    
def save_model(path, model, hist, optimizer, batch_size, learning_rate):
        # save model parameter
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'training_history': hist,
        }
        torch.save(state, path)

    
#%%     main

if __name__ == '__main__':
    log_name = input('Log name：')
    #device_no = input('Use device：')
    
    feature_extract = True
    num_classes = 7
    ignore_index = 6
    
    # hyperparameters
    batch_size = 32
    num_epochs = 40
    learning_rate = 2e-4
    
    # label weights()
    label_num = np.array([0.1666, 0.4746, 0.0857, 0.095, 0.0312, 0.2135, 1])
    
    var = 1/label_num
    weights = var/np.max(var) + 0.1
    weights[6] = 0
    weights = torch.tensor(weights).float()
    
    
    # Initialize the model for this run
    #model_ft = FCN32(num_classes=num_classes, feature_extract=feature_extract)
    model_ft = FCN16(num_classes=num_classes, feature_extract=feature_extract)

    # Print the model we just instantiated
    printw("model", model_ft)
    
    
    
    
    
    printw("Params to learn", "")
    params_to_update = model_ft.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
                
    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(params_to_update, lr=learning_rate)
    
    
    
    
    
    
    
    printw("Preparing dataLoader", " ")
    
    
    root_train = "/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p2_data/train"
    root_val = "/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p2_data/validation"
    dataset_train = P2Dataset(root=root_train, transform=transform_train)
    dataset_val = P2Dataset(root=root_val, transform=transform_val)
    
    dataLoader_train = torch.utils.data.DataLoader(dataset_train, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   num_workers=0)
    
    dataLoader_val = torch.utils.data.DataLoader(dataset_val, 
                                                   batch_size=batch_size, 
                                                   shuffle=True, 
                                                   num_workers=0)
    
    dataloaders_dict = {'train':dataLoader_train, 
                        'val':dataLoader_val}
    
    
    
    
    
    device = torch.device(("cuda") if torch.cuda.is_available() else "cpu")
    printw("Use device", device)
    
    
    
    
    
    
    
    
    
    # for testing model
    # dataIter_train = dataLoader_train.__iter__()
    # (x, label) = dataIter_train.next()
    # y_test = model_ft(x)
    
    
    
    # Setup the loss fxn
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=weights)
    
    
    model_ft = model_ft.to(device)
    
    # Train and evaluate
    model_ft, hist = train_model(model=model_ft, 
                                 dataloaders=dataloaders_dict, 
                                 criterion=criterion, 
                                 optimizer=optimizer_ft, 
                                 num_epochs=num_epochs)
    
    
    
    path_log = os.path.join("./log", log_name, 'log')
    save_model(path_log, model_ft, hist, optimizer_ft, batch_size, learning_rate)
    printw("Log path", path_log)
    
    '''
    plt.figure()
    plt.title("Training loss")
    plt.xlabel("Training Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,len(hist['training_loss'])+1),hist['training_loss'])
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Validation loss")
    plt.xlabel("Validation Epochs")
    plt.ylabel("Loss")
    plt.plot(range(1,len(hist['validation_loss'])+1),hist['validation_loss'])
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title("Validation Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel("Validation IoU")
    plt.plot(range(1,len(hist['val_iou_history'])+1),hist['val_iou_history'])
    plt.ylim((0,1.))
    plt.xticks(np.arange(1, num_epochs+1, 1.0))
    plt.legend()
    plt.show()
    '''