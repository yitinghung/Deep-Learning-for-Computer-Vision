import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import imageio
import random

random.seed(0)


def HUtoGrayTest(img_root, output_test_root, hu_min, hu_max):
    '''
        Transfer HU value(.npy) to grayscale image(.jpg)
    '''

    # Make train_JPG val_JPG output root folder
    os.makedirs(output_test_root, exist_ok=True)

    # Split data into train / val
    total = sorted(os.listdir(img_root))           # 1,116 case (32,665 images)

    ### MACOS
    if '.DS_Store' in total:
        idx = total.index('.DS_Store')
        total.pop(idx)

    print(f'Size of test set: {len(total)}')

    # Save grayscale JPG file

    for case in total:

        imgs = sorted(os.listdir(os.path.join(img_root, case)))
        for img in imgs:
            # Load .npy file
            hu_img = np.load(os.path.join(img_root, case, img))

            # 0 < HU < 2550   (leaving values only between 0 to 2550)
            hu_img[hu_img <= 0] = 0 
            hu_img[hu_img >= 2550] = 1000

            # Normalize to 0~1  -> to grayscale 0~255
            hu_img = hu_img-np.min(hu_img)
            hu_range = np.max(hu_img)-np.min(hu_img)
            hu_range = 2550
            hu_img = (hu_img/hu_range)*255

            
            imageio.imwrite(os.path.join(output_test_root, img.replace('npy', 'jpg')),np.uint8(hu_img))


def HUtoGray(args):
    '''
        Transfer HU value(.npy) to grayscale image(.jpg)
    '''
    img_root = args.root
    output_train_root = args.out_train_root
    output_val_root = args.out_val_root


    train_ratio = args.split_ratio
    hu_min = args.hu_min
    hu_max = args.hu_max


    # Make train_JPG val_JPG output root folder
    os.makedirs(output_train_root, exist_ok=True)
    os.makedirs(output_val_root, exist_ok=True)

    # Split data into train / val
    total = sorted(os.listdir(img_root))           # 1,116 case (32,665 images)

    ### MACOS
    if '.DS_Store' in total:
        idx = total.index('.DS_Store')
        total.pop(idx)

    train = random.sample(total, k=int(len(total)*train_ratio))
    val = []
    for t in total:
        if t not in train:
            val.append(t)

    #train = total[:int(len(total)*train_ratio)]    # 1,004 case
    #val = total[int(len(total)*train_ratio):]      # 112 case
    print(f'Size of train set: {len(train)}', f'Size of val set: {len(val)}')

    # load dataframe
    alldata_df = pd.read_csv('./skull/records_train.csv')

    # Save grayscale JPG file
    split = [train, val]
    train_case = []
    val_case = []

    for set in split:
        count = 0
        if set == train:
            print(f'Saving training set images...')
        else:
            print(f'Saving validation set images...')
        for case in set:
            if count % 100 == 0:
                print(f'Series {count}/{len(set)}')
            imgs = sorted(os.listdir(os.path.join(img_root, case)))
            for img in imgs:
                if set == train:
                    train_case.append(img[:-4])
                else:
                    val_case.append(img[:-4])
                # Load .npy file
                hu_img = np.load(os.path.join(img_root, case, img))

                # hu_min < HU < hu_max   (leaving values only between 0 to 2550)
                hu_img[hu_img <= hu_min] = hu_min     
                hu_img[hu_img >= hu_max] = hu_max

                # Normalize to 0~1  -> to grayscale 0~255
                hu_img = hu_img-np.min(hu_img)
                hu_range = np.max(hu_img)-np.min(hu_img)
                hu_img = (hu_img/hu_range)*255

                if set == train:
                    output_root = output_train_root
                else:
                    output_root = output_val_root

                output_dir = os.path.join(output_root, case)

                os.makedirs(output_dir, exist_ok=True)
                
                imageio.imwrite(os.path.join(output_dir, img.replace('npy', 'jpg')),np.uint8(hu_img))
            count += 1


    train_df = alldata_df[alldata_df['id'].isin(train_case)]
    valid_df = alldata_df[alldata_df['id'].isin(val_case)]
    train_df.to_csv(args.out_train_csv, index=False)
    valid_df.to_csv(args.out_val_csv, index=False)

    csvtoLabelTxt(args.out_train_csv, args.out_train_label, args.bbox_size)
    csvtoLabelTxt(args.out_val_csv, args.out_val_label, args.bbox_size)


def create_imgList_txt(img_dir):
    '''
        Create image path list
    '''

    with open(img_dir + '.txt', 'w') as txtfile:
        patient_list = os.listdir(img_dir)
        ### MACOS
        if '.DS_Store' in patient_list:
            idx = patient_list.index('.DS_Store')
            patient_list.pop(idx)

        patient_list.sort()
        patient_dir = [os.path.join(img_dir,patient) for patient in patient_list]

        for pt in patient_dir:
            img_list = os.listdir(pt)
            img_list.sort()
            filenames = [os.path.join(pt,img) for img in img_list]

            for fn in filenames:
                txtfile.write(fn)
                txtfile.write("\n")

        txtfile.close()

def toYOLOform(frameCoords, bbox_size, i_pair): # frameCoords : str type

    x, y = frameCoords.split(' ')[i_pair*2:i_pair*2+2]

    x1 = int(x)# - bbox_size/2
    y1 = int(y)# - bbox_size/2

    return x1/512., y1/512., bbox_size/512., bbox_size/512.


def csvtoLabelTxt(csvFile, save_dir, bbox_size):
    '''
        To create a txt file for each .npy data as a standard label for YOLOv5 training
        Format: each row [class_id, x1, y1, w,h] represents a fracture bbox.
    '''

    data_df = pd.read_csv(csvFile)

    os.makedirs(save_dir, exist_ok=True)

    for idx, id in enumerate(data_df['id']): 
        
        patient_dir = os.path.join(save_dir, id[:-9])
        os.makedirs(patient_dir, exist_ok=True)

        with open(os.path.join(patient_dir, str(id)+'.txt'), 'w') as txtfile:

            if data_df['label'][idx] == 1:
                # calculate num of fracture coord "pairs" in each .npy file
                n_frac = len(data_df['coords'][idx].split(' ')) // 2 

                for i in range(n_frac): # i: the i(th) fracture coords
                    x1, y1, w, h = toYOLOform(data_df['coords'][idx], bbox_size, i)
                    txtfile.write(str(0) + " " + str(x1) + " " + str(y1) + " " + str(w) + " " + str(h))
                    txtfile.write("\n") 
                txtfile.close()
            
            else:
                txtfile.close()



def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../skull/train', type=str, help='train root images path')
    # path to training data
    parser.add_argument('--out_train_root', default='./data/images/train', type=str, help='output path of training data')
    parser.add_argument('--out_train_csv', default='./data/train_label.csv', type=str, help='output path of training csv')
    parser.add_argument('--out_train_label', default='./data/labels/train', type=str, help='output path of training labels')
    # path to validating data
    parser.add_argument('--out_val_root', default='./data/images/val', type=str, help='output path of validation data')
    parser.add_argument('--out_val_csv', default='./data/valid_label.csv', type=str, help='output path of validating csv')
    parser.add_argument('--out_val_label', default='./data/labels/val', type=str, help='output path of validating labels')
    # path to testing data
    parser.add_argument('--test_root', default='../skull/test', type=str, help='test root images path')
    parser.add_argument('--out_test_root', default='./data/test', type=str, help='output path of testing data')
    #hyper parameters
    parser.add_argument('--split_ratio', default=0.9, type=float)
    parser.add_argument('--hu_min', default=-100, type=int)
    parser.add_argument('--hu_max', default=2550, type=int)
    parser.add_argument('--bbox_size', default=32, type=int)

    parser.add_argument('--only_test', action='store_true')

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    
    args = argument()
    HUtoGrayTest(args.test_root, args.out_test_root, args.hu_min, args.hu_max)

    if only_test:
        # Split data into Train / Val  &  Transfer HU value(.npy) to grayscale image(.jpg)
        HUtoGray(args)


