import torch

from transformers import BertTokenizer
from PIL import Image
import argparse

from models import caption
from datasets import coco, utils
from configuration import Config
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('-i', '--img_path', type=str, help='path to input image directory', required=True)
#parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
parser.add_argument('-o', '--output_path', type=str, help='path to output image directory', required=True)
args = parser.parse_args()

image_path = args.img_path
output_path = args.output_path
checkpoint_path = args.checkpoint

config = Config()


model = torch.load('p2/model.pth')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)


def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token     # [101, 0, 0, ...., 0]
    mask_template[:, 0] = False              # [F, T, T, ....., T]

    return caption_template, mask_template




@torch.no_grad()
def evaluate(image):
    model.eval()
    for i in range(config.max_position_embeddings - 1):      # 第一個已經是cls
        predictions = model(image, caption, cap_mask)
        predictions, attn_weight, feat_size = predictions[0][:, i, :], predictions[1], predictions[2]
        predicted_id = torch.argmax(predictions, axis=-1)
        #print("predicted_id:", predicted_id)

        if predicted_id[0] == 102:
            return caption, attn_weight, feat_size

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption, attn_weight, feat_size


def visualize_att(filename, seq, attn_weight, feat_size):
    '''Visualize attention matrix'''
    img = os.path.join(image_path, filename)
    img = np.array(Image.open(img))
    w, h, c = img.shape
    #print(w, h)
    fig = plt.figure(figsize=(16, 8))
    #fig.suptitle("Visualization of Attention", fontsize=24)
    #fig.add_axes()
    len_seq = len(seq)
    #print("len seq:", len_seq, "len att:", len(attn_weight))
    #print(np.ceil(len(seq)/2))
    ax = fig.add_subplot(2, np.ceil(len_seq/2)+1, 1)     # +1是因為還有第一張放原圖
    ax.set_title('<start>')
    ax.imshow(img)
    plt.axis('off')
    
    for i in range(len_seq): 
        attn_heatmap = np.array(attn_weight[i]).reshape(feat_size)
        attn_heatmap = cv2.resize(attn_heatmap, (h, w))
        ax = fig.add_subplot(2, np.ceil(len_seq/2)+1, i+2)
        ax.set_title(seq[i])
        orig_img = ax.imshow(img)
        ax.imshow(attn_heatmap, alpha=0.9, extent=orig_img.get_extent())
        plt.axis('off')
    #plt.show()
    
    plt.savefig(os.path.join(output_path, filename.replace('jpg', 'png')))


filenames = ['bike.jpg', 'girl.jpg', 'sheep.jpg', 'ski.jpg', 'umbrella.jpg']
for i in range(len(filenames)):
    image = Image.open(os.path.join(image_path, filenames[i]))
    image = coco.val_transform(image)        # (3, 199, 299)
    image = image.unsqueeze(0)               # (1, 3, 199, 299)

    caption, cap_mask = create_caption_and_mask(
    start_token, config.max_position_embeddings)

    output, attn_weight, feat_size = evaluate(image)
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    #result = tokenizer.decode(output[0], skip_special_tokens=True)
    print(result.capitalize())


    seq = result.capitalize()
    tokens = tokenizer.tokenize(seq)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    #img = os.path.join(image_path, filenames[i])
    #print(tokens)
    visualize_att(filenames[i], tokens, attn_weight[0], feat_size)