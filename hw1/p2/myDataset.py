from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

def default_loader(path):
    return Image.open(path)

def label_transform(mask):
    mask = (mask >= 128).astype(int)
    mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
    label = np.empty((512, 512))
    label[mask == 3] = 0  # (Cyan: 011) Urban land 
    label[mask == 6] = 1  # (Yellow: 110) Agriculture land 
    label[mask == 5] = 2  # (Purple: 101) Rangeland 
    label[mask == 2] = 3  # (Green: 010) Forest land 
    label[mask == 1] = 4  # (Blue: 001) Water 
    label[mask == 7] = 5  # (White: 111) Barren land 
    label[mask == 0] = 6  # (Black: 000) Unknown
    label[mask == 4] = 6  # (Red: 100) Unknown
    return Image.fromarray(label)

class myDataset(Dataset):
    def __init__(self, root, transform=None, 
                 label_transform=label_transform, 
                 loader=default_loader, prediction=False):
        
        self.root = root
        file_names = []
        for i in os.listdir(root):
            file_names.append(i)
                
        self.file_names = file_names
        
        self.transform = transform
        self.label_transform = label_transform
        self.loader = loader
        self.prediction = prediction

    def __getitem__(self, index):
        file_name = self.file_names[index]
        
        file_path = os.path.join(self.root, file_name)
        img = self.loader(file_path)
        
        label_path = os.path.join(self.root, file_name.replace('sat.jpg', 'mask.png'))
        
        if self.prediction ==False:
            label = self.loader(label_path)
            label = self.label_transform(np.array(label))
        else:
            label = Image.new('I', (512, 512))
        
        if self.transform is not None:
            img, label = self.transform(img, label)
        
        return img,label,file_name

    def __len__(self):
        return len(self.file_names)

# def imshow(img):
#     npimg = img.numpy()
#     plt.figure()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
    
def transform_testing(image, mask):
       
    # Random Resize and crop
    if np.random.random() > 0.5:
        resize = transforms.Resize(size=(560, 560))
        image = resize(image)
        mask = resize(mask)
    
        #  crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = transforms.functional.crop(image, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)

    # Random horizontal flipping
    if np.random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    # Random vertical flipping
    if np.random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.vflip(mask)

    # Transform to tensor
    image = transforms.functional.to_tensor(image)
    mask = transforms.functional.to_tensor(mask)
    
    # Normalize
    # image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    
    return image, mask

#if __name__ == '__main__':
    # root = "/home/yiting/Documents/DLCV/hw1/hw1-yitinghung/hw1_data/p2_data/train"
    
    # train_dataset = myDataset(root=root, transform=transform_testing)
    
    # l = []
    # for i in range(5):
    #     data = train_dataset[i]
    #     img = np.transpose(data[1], (1, 2, 0))
    #     #plt.imshow(img)
    #     #plt.show()        
    #     l.append(data[1])
        
    
    