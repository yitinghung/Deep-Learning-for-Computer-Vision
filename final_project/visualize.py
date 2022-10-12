import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio

def Rectangle(ax, x, y):
    '''Visualize skull fracture. Bounding box 16*16. x, y: centroid'''

    ax.add_patch(
     patches.Rectangle(
        (x-8, y-8),
        16,
        16,
        edgecolor = 'red',
        fill=False      
     ) )


def save_vis(csv, img_root, output):
    '''Save visualization results. (Only save fractured imgs)'''

    if not os.path.exists(output):
        os.mkdir(output)

    df = pd.read_csv(csv, index_col='id').sort_values(['label'], ascending=False)

    print('Saving visualization result...')
    l = 0
    while(df['label'][l] == 1):
        if l % 100 == 0:
            print(f'save images {l}/{len(df)}')
        # Get image path & bounding box coordinates
        id = df.index[l]
        series_dir = id.rsplit('_', 1)[0]
        img_dir = os.path.join(img_root, series_dir, f'{id}.npy')
        coords = df['coords'][l]

        # Load .npy file
        hu_img = np.load(img_dir)

        # -300 < HU < 1000   (leaving values only between -300 to 1000)
        hu_img[hu_img<=-300] = -300   # -300 or -500  (?)        
        hu_img[hu_img>=1000] = 1000

        # Normalize to 0~1  -> to grayscale 0~255
        hu_img = hu_img-np.min(hu_img)
        hu_range = np.max(hu_img)-np.min(hu_img)
        hu_img = (hu_img/hu_range)*255

        # Visualize bounding box
        plt.figure()
        plt.imshow(hu_img, cmap='gray')
        ax = plt.subplot()
        for i in range(0, len(coords.split(' ')), 2):
            x, y = int(coords.split(' ')[i]), int(coords.split(' ')[i+1])
            Rectangle(ax, x, y)

        # Output images with bounding box
        output_pth = os.path.join(output, series_dir)
        if not os.path.exists(output_pth):
            os.mkdir(output_pth)
        plt.savefig(os.path.join(output_pth,f'{id}.jpg'))

        #plt.show()
        l += 1



def img_show(img_dir, csv):
    '''Show all images of single case'''

    files = [file for file in os.listdir(img_dir)]
    files.sort()
    for i in range(len(files)):
        img_pth = os.path.join(img_dir, files[i])

        # Load .npy file 
        hu_img = np.load(img_pth)

        # Set HU range [0, 2550]
        hu_img[hu_img<=0] = 0   # Range?? to be decided          
        hu_img[hu_img>=2550] = 2550
        # hu_img[hu_img<=-300] = -300   # Range?? to be decided          
        # hu_img[hu_img>=1000] = 1000

        # Normalize to 0~1  -> to grayscale 0~255
        hu_img = hu_img-np.min(hu_img)
        hu_range = np.max(hu_img)-np.min(hu_img)
        hu_img = (hu_img/hu_range)*255
        
        # plot original image
        plt.subplot(1, 2, 1) 
        plt.imshow(hu_img, cmap='gray')
        plt.title(f'{files[i]}')

        # plot image with bounding box 
        ax = plt.subplot(1, 2, 2)
        plt.imshow(hu_img, cmap='gray')

        # Get box coords
        df = pd.read_csv(csv).fillna(0)
        id = files[i].split('.')[0]
        coords = df['coords'][df.index[df['id'] == id]].values[0]

        # If there exits skull fractures, plot the box
        if type(coords) == str:
            print(coords)
            for i in range(0, len(coords.split(' ')), 2):
                x, y = int(coords.split(' ')[i]), int(coords.split(' ')[i+1])
                Rectangle(ax, x, y)

        #plt.savefig("test.png")
        plt.show()



if __name__ == '__main__':
    csv_pth = './skull/records_train.csv'
    img_root = './skull/train'
    output_dir = './skull/Vis_data/'
    # Save all visualization results of skull fracture images. 
    save_vis(csv=csv_pth, img_root=img_root, output=output_dir)

    # Select a series(patient) directory for visualization (show image but not save)
    img_dir = './skull/train/H1_00000005_00000490'
    img_show(img_dir=img_dir, csv=csv_pth)






# HU transformation reference:
# 0 ~ 2550         http://proceedings.mlr.press/v121/kuang20a/kuang20a.pdf
# -300 ~ 1000      https://hannes.nickisch.org/students/vankersbergen21master.pdf

