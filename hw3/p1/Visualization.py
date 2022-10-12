import torch
import torch.nn as nn
import torchvision.transforms as transforms
from Dataset import myDataset
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import random
import os
import timm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def Vis_pos_emb(model):
    pos_embed = model.pos_embed
    print(pos_embed.shape)
    # Visualize position embedding similarities.
    # One cell shows cos similarity between an embedding and all the other embeddings.
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle("Visualization of position embedding similarities", fontsize=24)
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((14, 14)).detach().cpu().numpy()
        ax = fig.add_subplot(14, 14, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)
    plt.savefig('pos_emb.png')

def Vis_att_mat(fn, model):
    img = Image.open(fn)
    print("Image size:", img.size) # (375, 500)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    img_tensor = transform(img).unsqueeze(0)

    patches = model.patch_embed(img_tensor)     # patch embedding convolution
    print("Image tensor: ", img_tensor.shape)
    print("Patch embeddings: ", patches.shape)

    pos_embed = model.pos_embed
    print("Position embeddings: ", pos_embed.shape)

    transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
    print("Transformer input: ", transformer_input.shape)

    print("Transformer Multi-head Attention block:")
    attention = model.blocks[11].attn
    print(attention)
    print("input of the transformer encoder:", transformer_input.shape)

    # fc layer to expand the dimension
    transformer_input_expanded = attention.qkv(transformer_input)[0]
    print("expanded to: ", transformer_input_expanded.shape)

    # Split qkv into mulitple q, k, and v vectors for multi-head attantion
    qkv = transformer_input_expanded.reshape(197, 3, 12, 64)  # (N=197, (qkv), H=12, D/H=64)
    print("split qkv : ", qkv.shape)
    q = qkv[:, 0].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
    # class_q = q[:, 0]
    # print('class q shape:', class_q.shape)
    print("q shape:", q.shape)
    k = qkv[:, 1].permute(1, 0, 2)  # (H=12, N=197, D/H=64)
    kT = k.permute(0, 2, 1)  # (H=12, D/H=64, N=197)
    print("transposed ks: ", kT.shape)

    # Attention Matrix
    attention_matrix = q @ kT
    print("attention matrix: ", attention_matrix.shape)
    #plt.imshow(attention_matrix[11].detach().cpu().numpy())

    # Visualize attention matrix
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Visualization of Attention", fontsize=24)
    fig.add_axes()
    img = np.asarray(img)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(img)


    attention_matrix = torch.mean(attention_matrix, dim=0)
    print("mean attention matrix:", attention_matrix.shape)
    attention_matrix = torch.mean(attention_matrix, dim=0)
    attn_heatmap = attention_matrix[1:].reshape((14, 14)).detach().cpu().numpy()
    # attn_heatmap = attention_matrix[0, 1:].reshape((14, 14)).detach().cpu().numpy()
    # print(attn_heatmap.shape)
    resize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        ])
    attn_heatmap = resize(attn_heatmap)
    attn_heatmap = np.array(attn_heatmap)
    ax = fig.add_subplot(1, 2, 2)
    orig_img = ax.imshow(img)
    ax.imshow(attn_heatmap, alpha=0.8, extent=orig_img.get_extent())
    #plt.savefig("test_img3.png") 
    #ax = fig.add_subplot(1, 2, 2)
    #ax.imshow(attn_heatmap)
    plt.show()

if __name__ == '__main__':
    # Load model for visualization 
    #ckpt_pth = 'checkpoints/model.pth'
    ckpt_pth =  'p1/model.pth'
    checkpoints = torch.load(ckpt_pth, map_location='cpu')
    vis_img1 = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/val/26_5064.jpg'
    vis_img2 = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/val/29_4718.jpg'
    vis_img3 = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/val/31_4838.jpg'
    vis_img4 = '/home/yiting/Documents/DLCV/hw3/hw3_data/hw3_data/p1_data/val/31_4993.jpg'

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=37)
    model.load_state_dict(checkpoints['state_dict'])
    #Vis_pos_emb(model)
    Vis_att_mat(vis_img4, model)

# reference: https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb#scrollTo=j3tQ2rAzX1gf