import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import imageio

def HUtoGray(img_root, train_ratio, output_train_root, output_val_root):
    '''Transfer HU value(.npy) to grayscale image(.jpg)'''

    # Make train_JPG val_JPG output root folder
    if not os.path.exists(output_train_root):
        os.mkdir(output_train_root)
    if not os.path.exists(output_val_root):
        os.mkdir(output_val_root)

    # Split data into train / val
    total = sorted(os.listdir(img_root))           # 1,116 case (32,665 images)
    train = total[:int(len(total)*train_ratio)]    # 1,004 case
    val = total[int(len(total)*train_ratio):]     # 112 case
    print(f'Size of train set: {len(train)}', f'Size of val set: {len(val)}')

    # Save grayscale JPG file
    split = [train, val]

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
                # Load .npy file
                hu_img = np.load(os.path.join(img_root, case, img))

                # 0 < HU < 2550   (leaving values only between 0 to 2550)
                hu_img[hu_img<=0] = 0      
                hu_img[hu_img>=2550] = 2550

                # Normalize to 0~1  -> to grayscale 0~255
                hu_img = hu_img-np.min(hu_img)
                hu_range = np.max(hu_img)-np.min(hu_img)
                hu_img = (hu_img/hu_range)*255

                if set == train:
                    output_root = output_train_root
                else:
                    output_root = output_val_root

                output_dir = os.path.join(output_root, case)
                if not os.path.exists(output_dir):
                    os.mkdir(output_dir)
                
                imageio.imwrite(os.path.join(output_dir, img.replace('npy', 'jpg')),np.uint8(hu_img))
            count += 1

def create_imgList_txt(img_dir):
    '''Create image path list'''

    with open(img_dir + '.txt', 'w') as txtfile:
        patient_list = os.listdir(img_dir)
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

    x1 = int(x) - bbox_size/2
    y1 = int(y) - bbox_size/2

    return x1/512., y1/512., bbox_size/512., bbox_size/512.


def csvtoLabelTxt(csvFile, save_dir, bbox_size):
    '''to create a txt file for each .npy data as a standard label for YOLOv5 training
       Format: each row [class_id, x1, y1, w,h] represents a fracture bbox.'''

    data_df = pd.read_csv(csvFile)

    if os.path.isdir(save_dir) == False:
        os.mkdir(save_dir)

    for idx, id in enumerate(data_df['id']): 
        
        patient_dir = os.path.join(save_dir, id[:-9])
        if os.path.isdir(patient_dir) == False:
            os.mkdir(patient_dir)

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


def splitCSV(csvFile, train_ratio, train_dir):
    alldata = pd.read_csv(csvFile)
    alldata_df = pd.DataFrame(alldata)

    patient_list = os.listdir(train_dir)
    patient_list.sort()
    valid_patient = patient_list[int(len(patient_list)*train_ratio)]

    valid_frame_list = os.listdir(os.path.join(train_dir,valid_patient))
    valid_frame_list.sort()
    valid_frame_name = valid_frame_list[0][:-4]

    valid_idx = alldata_df.loc[alldata_df['id'] == valid_frame_name].index.to_numpy()

    train_df = alldata_df[:int(valid_idx)]
    valid_df = alldata_df[int(valid_idx):]

    train_df.to_csv('./skull/train_label.csv', index=False)
    valid_df.to_csv('./skull/valid_label.csv', index=False)




if __name__ == '__main__': 
    img_root = './skull/train'
    output_train_root = './skull/train_JPG'
    output_val_root = './skull/val_JPG'
    train_ratio = 0.9

    # Split data into Train / Val  &  Transfer HU value(.npy) to grayscale image(.jpg)
    HUtoGray(img_root, train_ratio, output_train_root, output_val_root)

    # Create Train / Val image path list (.txt)
    create_imgList_txt(output_train_root)
    create_imgList_txt(output_val_root)

    
    csvFile = './skull/records_train.csv' # csv file path

    # Split records_train.csv(total) file into train_JPG.csv & val_JPG.csv
    splitCSV(csvFile, train_ratio, img_root)

    
    train_csv = './skull/train_label.csv'
    valid_csv = './skull/valid_label.csv'
    train_label_dir = './skull/train_label_YOLO'
    valid_label_dir = './skull/valid_label_YOLO'
    bbox_size = 16  #(bbox_h = bbox_w = 16)

    # Create Label file (YOLO format) 
    csvtoLabelTxt(train_csv, train_label_dir, bbox_size) 
    csvtoLabelTxt(valid_csv, valid_label_dir, bbox_size) 
