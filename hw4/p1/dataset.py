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

import matplotlib.pyplot as plt
import numpy as np

filenameToPILImage = lambda x: Image.open(x)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, data_dir, csv_path):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
        self.filenames = self.data_df["filename"].tolist()
        self.labels = self.data_df["label"].tolist()

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]
        image = self.transform(os.path.join(self.data_dir, filename))
        return image, label

    def __len__(self):
        return len(self.data_df)
