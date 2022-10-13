# DLCV HW3
## Description
### Problem 1: Image Classification with Vision Transformer
* Train a Vision Transformer for image classification.      
&emsp;&emsp;&emsp;Input: RGB image     
&emsp;&emsp;&emsp;Ouput: Classification label      
&emsp;&emsp;&emsp;Evaluation metric: Accuracy
* Visualize position embeddings.
* Visualize attention map.

### Problem 2: Visualization of Attention in Image Captioning
* Analyze the transformer decoder in image captioning by visualizing the cross-attention between images and generated captions.
* Given an input image, your model would be able to generate a corresponding caption sequentially, and you have to visualize the cross-attention between the image patches and each predicted word in your own caption.

## Usage
```
git clone https://github.com/yitinghung/Deep-Learning-for-Computer-Vision.git
cd hw3
```
## Dataset
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `hw3_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1PDlObdTW6eLJiencXM5OdkSTFVSNvoOl/view?usp=sharing) and unzip the compressed file manually.    

## Install Packages
```
pip3 install -r requirements.txt
```

## Implementations
### Problem 1: Image Classification with Vision Transformer
```
cd p1/
```
#### Train
```
python3 Train.py
```

#### Inference
```
bash hw3_1.sh <test_dataset_dir> <output_csv>
```
The output csv will be in two columns with column names ["filename", "label"]

#### Evaluation
```
python3 cal_acc.py
```

#### Visualization
```
python3 Visualization.py
``` 
Visualize the attention map between the [class] token (as query vector) and all patches (as key vectors) from the LAST multi-head attention layer


### Problem 2: Visualization of Attention in Image Captioning
```
cd p2/
```
#### Pretrain
Pretrained model and weights refer to the [github](https://github.com/saahiluppal/catr)

#### Inference & Visualization
```
bash hw3_2.sh <test_images_dir> <output_visualization>
```



