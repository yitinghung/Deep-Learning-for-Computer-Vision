# DLCV HW1
## Description
### Problem 1: Image Classification
Perform image classification by training a CNN model to predict a label for each image.
* Input: RGB images
* Output: Classification label
* Model Evaluation: Accuracy
### Problem 2: Semantic Segmentation
Perform semantic segmentation by training a CNN model to predict the label for each pixel in an image.
* Input : RGB image
* Output : Semantic segmentation/prediction
* Model Evaluation: mean Intersection over Union (mIoU)

## Usage
```
git clone https://github.com/yitinghung/Deep-Learning-for-Computer-Vision.git
cd hw1
```

## Dataset
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `hw1_data`. Note that this command by default only works on Linux. If you are using other operating systems or you can not download the dataset by running the command above, you should download the dataset from [this link](https://drive.google.com/file/d/1CIvfO8rDMq5-3vmi0c6mrhDfzqrAZDem/view?usp=sharing) and unzip the compressed file manually.

## Packages
```
pip3 install -r requirements.txt
```
Note that using packages with different versions will very likely lead to compatibility issues.

## Implemetations
### Problem 1 - Image Classification
```
cd p1/
```

#### Train
```
python3 p1_train.py
```

#### Inference
```
bash hw1_1.sh <--test_dataset_dir> <--output_csv>
```


### Problem 2 - Semantic Segmentation
#### Train
```
python3 p2_train.py
```

#### Inference
```
bash hw1_2.sh <--test_dataset_dir> <--output_segmented_image_dir>
```

#### Evaluation
```
python3 mean_iou_evaluate.py <-g ground_truth_directory> <-p prediction_directory>
```
Note that the predicted segmentation semantic map file should have the same filename as that of its corresponding ground truth label file (both of extension ``.png``).

#### Visualization
To visualization the ground truth or predicted semantic segmentation map in an image, you can run the provided visualization script provided in the starter code by using the following command.     
```
python3 viz_mask.py <--img_path xxxx_sat.jpg> <--seg_path xxxx_mask.png>
```

