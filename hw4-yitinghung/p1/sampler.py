import torch
import numpy as np
from dataset import MiniDataset
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision


class CategoriesSampler():

	def __init__(self, n_batch, total_cls , n_way , n_img, n_shot):
		self.n_batch = n_batch
		self.total_cls = total_cls
		self.n_way = n_way
		self.n_img = n_img
		self.classes = 0

		self.iters = []     # 裝每5 way各是哪一個類別 n_batch * batch_size = 500 * 5 
		batch_size = n_way  # N way
		while len(self.iters) < n_batch:
			self.classes = np.arange(total_cls)
			np.random.shuffle(self.classes)         # random sample class 
			for i in range (total_cls // batch_size):
				self.iters.append(self.classes[i * batch_size: (i + 1) * batch_size] )
				if len(self.iters) == n_batch:    # for迴圈裡就到batch size，直接break
					break                    

	def __iter__(self):
		for self.classes in self.iters:
			batch =[]
			for one_class in self.classes:

				img_idx = np.random.randint(0, 600, self.n_img)
				img_idx = one_class * 600 + img_idx
				
				batch.append(torch.tensor(img_idx, dtype=torch.int))

			batch = torch.stack(batch).t().reshape(-1)
			yield batch
				
	def __len__(self):
		return self.n_batch   





