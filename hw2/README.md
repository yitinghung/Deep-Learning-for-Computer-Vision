# DLCV HW2
This repository implement the models of both GAN and ACGAN for generating human face images and digits, respectively, and the model of DANN for classifying digit images from different domains.
## Description
### Problem 1: Image Generation and Feature Disentanglement - GAN
* A generative adversarial network (GAN) is a deep learning method in which two neural networks (Generator and Discriminator) compete with each other and improve themselves together.
* Implement a GAN model from scratch and train it on the face dataset.
* Evaluation:    
1. Fréchet inception distance (FID)
2. Inception score (IS)
<img src="https://github.com/yitinghung/Deep-Learning-for-Computer-Vision/blob/main/hw2/p1_figure.png" width=50% height=50%>

### Problem 2: Image Generation and Feature Disentanglement - ACGAN
* Auxiliary Classifier GAN (ACGAN) is a conditional GAN method applying auxiliary classifers for conditional image synthesis and feature disentanglement.
* Aim to generate images for the corresponding digit inputs
* Implement an ACGAN model from scratch and train it on the mnistm dataset
* Evaluation:   
Evaluate the generated output by the classification accuracy with a pretrained digit classifier
<img src="https://github.com/yitinghung/Deep-Learning-for-Computer-Vision/blob/main/hw2/p2_figure.png" width=50% height=50%>

### Problem 3: Unsupervised Domain Adaptation (UDA) - DANN
* Implement DANN on digits datasets (USPS, MNIST-M and SVHN) and consider the following 3 scenarios:    
SVHN → MNIST-M, MNIST-M → USPS, USPS → SVHN   (Source domain → Target domain)
\*Note that during training DANN, we utilize the images and labels of source domain, and only images (without labels) of target domain.
* Evaluation:
Accuracy on target domain
<img src="https://github.com/yitinghung/Deep-Learning-for-Computer-Vision/blob/main/hw2/p3_figure.png" width=50% height=50%>

## Usage
```
git clone https://github.com/yitinghung/Deep-Learning-for-Computer-Vision.git
cd hw2/
```

## Dataset
```
bash ./get_dataset.sh
```
The shell script will automatically download the dataset and store the data in a folder called `hw2_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1BwZiFfGKAqIOFRupt6xO7-KuhPYd5VMO/view?usp=sharing) and unzip the compressed file manually.

## Packages
```
pip3 install -r requirements.txt
```

## Implementations
### Problem 1: Image Generation and Feature Disentanglement - GAN
``` 
cd p1/
```

#### Train
```
python3 P1_Train.py
```

#### Inference 
```
bash hw2_p1.sh <generated_test_data_path>
```

### Problem 2: Image Generation and Feature Disentanglement - ACGAN
```
cd p2/
```

#### Train
```
python3 P2_Train.py
```

#### Inference
```
bash hw2_p2.sh <generated_test_data_path>
```

### Problem 3: Unsupervised Domain Adaptation (UDA) - DANN
```
cd p3/
```

#### Train
```
python3 P3_Train_DANN.py
```

#### Inference
```
bash hw2_p3.sh <target_data_pth> <TARGET> <output_csv_pth>
```

#### Evaluation
```
python3 ../hw2_eval.py <output_csv_pth> <gt_csv_pth>
```
Note that for `hw2_eval.py` to work, your predicted `.csv` files should have the same format as the ground truth files we provided in the dataset as shown below.
| image_name | label |
|:----------:|:-----:|
| 00000.png  | 4     |
| 00001.png  | 3     |
| 00002.png  | 5     |
| ...        | ...   |

