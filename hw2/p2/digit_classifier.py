import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import random
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import argparse

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



def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class ClassifierDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = [file for file in os.listdir(root)]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __getitem__(self, index):
        filepath = os.path.join(self.root, self.files[index])
        img = Image.open(filepath)
        img = self.transform(img)
        label = int(self.files[index].split('_')[0])
        filename = self.files[index]
        return img, label, filename

    def __len__(self):
        return len(self.files)


def Test(model, pred_save_pth, test_loader):
    model = model.to(device)
    model.device = device

    model.eval()
    filenames = []
    predictions = []
    test_acc = []

    for i, data in enumerate(test_loader):
        imgs, labels, filename = data
        imgs, labels = imgs.to(device), labels.to(device)
        print(imgs.shape)
        with torch.no_grad():
            logits = model(imgs)

        acc = (logits.argmax(dim=-1) == labels).float().mean()
        test_acc.append(acc)

        filenames.extend(filename)
        predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        
    dic = {'real_label': filenames, 'predict_label': predictions}
    df = pd.DataFrame(dic)
    df.to_csv(pred_save_pth, index=0)

    test_acc = sum(test_acc) / len(test_acc)
    print(f'[ Accuracy ] = {test_acc:.5f}')

    torch.cuda.empty_cache() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--images', help='images_path', type=str)
    args = parser.parse_args()
    
    same_seeds(2021)
    batch_size = 128

    # load digit classifier
    net = Classifier()
    path = "p2/Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)

    print(net)

    root = args.images
    dataset = ClassifierDataset(root)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    output_pth = 'prediction.csv'
    Test(net, output_pth, dataloader)


