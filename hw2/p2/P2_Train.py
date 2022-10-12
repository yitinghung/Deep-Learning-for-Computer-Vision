import torch
from torch import nn
import torch.utils.data
import torchvision.transforms as transforms
from P2_Model import Generator, Discriminator
from P2_Dataset import P2Dataset
from torch.utils.data import Dataset,DataLoader
from torch import optim
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image
import random
import os
from torch.autograd import Variable
from utils import weights_init, compute_acc


img_size = 32 
batch_size = 128
channels = 3
latent_dim = 100
#ngf = 64
#ndf = 64
n_epochs = 200
lr = 0.0002
b1 = 0.5
b2 = 0.999
n_classes = 10
sample_interval = 400 

ckpt_outputdir = './checkpoints'
img_outputdir = './generated_imgs'


if not os.path.exists(ckpt_outputdir):
    os.makedirs(ckpt_outputdir)
if not os.path.exists(img_outputdir):
    os.makedirs(img_outputdir)

# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator(n_classes, latent_dim, img_size, channels)
discriminator = Discriminator(channels, img_size, n_classes)

cuda = True if torch.cuda.is_available() else False
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()


# Initialize weights
generator.apply(weights_init)
discriminator.apply(weights_init)

# Configure data loader
root = '/home/yiting/Documents/DLCV/hw2/hw2-yitinghung/hw2_data/digits/mnistm/train'
csv_pth = '/home/yiting/Documents/DLCV/hw2/hw2-yitinghung/hw2_data/digits/mnistm/train.csv'
transform = transforms.Compose([transforms.Resize(img_size),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = P2Dataset(root, csv_pth, transform = transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, fn):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, os.path.join(img_outputdir, "%s.png" % fn), nrow=n_row, normalize=True)


# ----------
#  Training
# ----------

for epoch in range(n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity, pred_label = discriminator(gen_imgs)
        g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch+1, n_epochs, i+1, len(dataloader), d_loss.item(), 100 * d_acc, g_loss.item())
        )
        #batches_done = epoch * len(dataloader) + i
        #if batches_done % sample_interval == 0:
    sample_image(n_row=10, fn = f'{epoch+1}')

    # do checkpointing
    torch.save(generator.state_dict(), f'{ckpt_outputdir}/generator_epoch_{epoch+1}.pth')
    torch.save(discriminator.state_dict(), f'{ckpt_outputdir}/discriminator_epoch_{epoch+1}.pth')