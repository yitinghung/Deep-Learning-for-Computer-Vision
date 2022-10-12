from P2_Model import Generator
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch
from torchvision.utils import save_image
import os
import numpy as np
import random
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

def sample_image(n_classes, fn, latent_dim):
    """Generate 100 images for evaluation (ranging from 0 to n_classes)"""
    # Sample noise
    z = Variable(torch.FloatTensor(np.random.normal(0, 1, (n_classes ** 2, latent_dim))).to(device))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for i in range(n_classes) for num in range(n_classes)])
    labels = Variable(torch.LongTensor(labels).to(device))
    fake_imgs = generator(z, labels)
    save_image(fake_imgs.data, "%s.png" % fn, nrow=n_classes, normalize=True)


if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', help='output_path', type=str)
    args = parser.parse_args()

    print(f'output_path: {args.output_path}')
    output_path = args.output_path
    checkpoint_pth = 'p2/p2_model.pth'

    same_seeds(0)
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print("Using Device:", device)
    channels = 3
    latent_dim = 100
    batchSize = 100
    n_classes = 10
    img_size = 32

    for digit in range(n_classes):
        # Generate 100 images for each digit
        # Sample noise
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batchSize, latent_dim))).to(device))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([digit for i in range(batchSize)])
        labels = Variable(torch.LongTensor(labels).to(device))
   

        # Create the generator
        generator = Generator(n_classes, latent_dim, img_size, channels).to(device)        # Load checkpoint
        checkpoint = torch.load(checkpoint_pth,  map_location=device)
        generator.load_state_dict(checkpoint)
        print('model loaded!')


        generator.eval()
        with torch.no_grad():
            fake = generator(z, labels).detach().to(device) 
            fake = (1+fake)/2

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=28),
                transforms.ToTensor()
            ])

            L = [transform(f) for f in fake]
            #print('image size:', len(L[0][0]))
            for i in range(len(L)):
                save_image(L[i], os.path.join(output_path, f'{digit}_{i+1:03d}.png'))

            print(f'image of digit {digit} saved!')
    
    #sample_image(n_classes=10, fn='output100', latent_dim=100)
