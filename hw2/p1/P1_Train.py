import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as utils
from P1_Model import Discriminator,Generator
from P1_DataLoader import P1Dataset
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
import os
from utils import save_checkpoint, weights_init 

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


def train(netG, netD, epoch, train_loader, device, batch_size, nz, lr, ckpt_pth, generate_img_pth):
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    #% Set training mode
    netG.train()
    #% Binary cross entropy loss
    criterion = nn.BCELoss()
    #% Create batch of latent vectors that we will use to visualize the progression of the generator    
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)  # 確認一下nz到底是什麼

    #% Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    #%% Training Loop
    print(f'Start Training Loop. Using device:{device}.')

    #% Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    for ep in range(epoch):
        for i, data in enumerate(train_loader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ############################
            #%% Train with all-real batch
            netD.zero_grad()
            #% Format batch
            data = data.to(device)
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            #% Forward pass real batch through D
            output = netD(data).view(-1)

            #% Calculate loss on all-real batch
            errD_real = criterion(output, label)
            #% Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            #%% Train with all-fake batch
            #% Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            #% Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            #% Classify all fake batch with D
            output = netD(fake.detach()).view(-1)  # 弄清楚detach用法

            #% Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            #% Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            #% Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            #% Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            #% Since we just updated D, perform another forward pass of all-fake batch through D
            output = netD(fake).view(-1)
            #% Calculate G's loss based on this output
            errG = criterion(output, label)
            #% Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            #% Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print(f'[Epoch {ep+1}/{epoch}][Batch {i+1}/{len(train_loader)}]\tLoss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\tD(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')
            
            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((ep == epoch-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                images = utils.make_grid(fake, padding=2, normalize=True)
                img_list.append(images)
                imageio.imwrite(os.path.join(generate_img_pth, f'{iters}.png'), np.transpose(images, (1, 2, 0)))
           
            iters += 1

        # save_checkpoint(os.path.join(ckpt_pth, f'G-{iters}.pth'), netG, optimizerG)
        torch.save(netG, os.path.join(ckpt_pth, f'finalG-{D_G_z2:.5f}.pth'))
        # save_checkpoint(os.path.join(ckpt_pth, f'D-{iters:.5f}.pth'), netD, optimizerD)
        torch.save(netD,os.path.join(ckpt_pth, f'finalD-{D_G_z2:.5f}.pth'))
        # torch.save({
		# 'generator': netG.state_dict(),
		# 'discriminator': netD.state_dict(),
		# 'optimizerG': optimizerG.state_dict(),
		# 'optimizerD': optimizerD.state_dict(),
	    # }, f'{ckpt_pth}/model_epoch_{ep}.pth')


    #% Plot D & G’s losses versus training iterations
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('loss.png')
    plt.show()   

    #% Visualization of G’s progression
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i,(1, 2, 0)))] for i in img_list]
    plt.savefig('G_progression.png')
    plt.show()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(train_loader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(utils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('real_fake.png')
    plt.show()

if __name__ == '__main__':
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    same_seeds(2021)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batchsize = 128
    epoch = 500
    lr = 0.0002 


    root = '/home/yiting/Documents/DLCV/hw2/hw2-yitinghung/hw2_data/face/train'
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = P1Dataset(root=root, transform=tfm)
    trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
    ckpt_outputdir = './checkpoints'
    img_outputdir = './generate_imgs'

    if not os.path.exists(ckpt_outputdir):
        os.makedirs(ckpt_outputdir)
    if not os.path.exists(img_outputdir):
        os.makedirs(img_outputdir)

    # Check trainloader dimension
    print('Trainset length:', len(trainset))
    dataiter = iter(trainloader)
    images = dataiter.next()
    print('Image tensor in each batch:', images.shape, images.dtype)

    # Number of channels in the training images(input). For color images this is 3
    nc = 3
    # Size of z latent vector (i.e. size of generator input)
    nz = 1024
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64

    # Create the generator
    netG = Generator(nz, ngf, nc).to(device)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netG.apply(weights_init)  #注意print出來的東西
    print(netG)

    # Create the discriminator
    netD = Discriminator(nc, ndf).to(device)
    # Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
    netD.apply(weights_init)
    print(netD)

    train(netG, netD, epoch, trainloader, device, batchsize, nz, lr, ckpt_pth=ckpt_outputdir, generate_img_pth=img_outputdir)