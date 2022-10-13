# DLCV HW4
## Description
### Problem 1: Few-Shot Learning - Prototypical Network 
* Prototypical Networks learn a metric space in which classification can be performed by computing distances to prototype representations of each class.
* Implement the prototypical network as your baseline model to perform 5-way K-shot classification. (K=1, 5, 10)
* Model Evaluation: Report the 95% confidence interval of the accuracy over 600 episodes. In each episode, the accuracy is calculated over totally N * 15 query data where each way has 15 query data (N-way K-shot setting).
$$ \bar{x}\pm1.96x\frac{\sigma}{\sqrt{600}}$$

### Problem 2: Self-Supervised Pre-training for Image Classification
* Pre-train ResNet50 backbone on Mini-ImageNet via the recently self-supervised learning methods (this repo use [BYOL](https://github.com/lucidrains/byol-pytorch)). 
* After pretraining, you can conduct downstream task (i.e., image classification) with different settings to analyze your pre-trained backbone. 

## Usage
```
git clone https://github.com/yitinghung/Deep-Learning-for-Computer-Vision.git
cd hw4/
```

## Dataset
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `hw4_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/drive/folders/19KSzhJyGjEkh-pKUds26LwN9IQ5SUSOf?usp=sharing) and unzip the compressed file manually.

## Install Packages
```
pip3 install -r requirements.txt
```

## Implementations
### Problem 1: Few-Shot Learning - Prototypical Network
#### Train
```
python train.py
```
#### Download Checkpoints
```
bash ../hw4_download.sh
```
#### Inference
```
bash hw4_p1.sh <test_csv> <test_dataset_dir> <test_case_csv> <output_csv>
```
#### Evaluation
```
python eval.py <output_csv> <ground_truth_csv>
```

### Problem 2: Self-Supervised Pre-training for Image Classification
#### Train
Training using BYOL
```
train_BYOL.py
```
compare with supervised learning
```
train_finetune.py
```
#### Download Checkpoints
```
bash ../hw4_download.sh
```
#### Inference
```
bash hw4_p2.sh <test_csv> <test_dataset_dir> <output_csv>
```




