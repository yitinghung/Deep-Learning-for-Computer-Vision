# DLCV Final Project ( Skull Fractracture Detection )

# How to Run
    git clone https://github.com/ultralytics/yolov5.git
     
    python3 inf_preprocessing.py --only_test --test_root <Raw TESTING DATA PATH> --out_test_root ./data/test
     
    python3 yolov5/detect.py --source ./data/test/ --img 512 --max-det 6 --weights ./ckpt/yolov5s_adamw.pt --conf-thres 0.5 --save-txt --project ./result --name adamw
    python3 yolov5/detect.py --source ./data/test/ --img 512 --max-det 6 --weights ./ckpt/yolov5s_sgd.pt --conf-thres 0.5 --save-txt --project ./result --name sgd
     
    python3 ensemble_inference.py --exp_path_1 ./result/adamw --exp_path_2 ./result/sgd --output_csv ./result.csv


### Dataset
    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `skull`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1i2MlS-eAkx0bFFKirSEmQyp5_FIPJO7p/view?fbclid=IwAR3-xGO3EOTQBoTR_PtCAlHIVK_QxMz-WmzoiZrSC8PWsdM1k0xGU5HW6vg) and unzip the compressed file manually.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Evaluation Code
In the starter code of this repository, we have provided a python script for evaluating the results for this project. For Linux users, use the following command to evaluate the results.
```bash
python3 for_students_eval.py --pred_file <path to your prediction csv file> --gt_file <path to the ground-truth csv file>
```
![image](https://github.com/yitinghung/Deep-Learning-for-Computer-Vision/blob/main/final_project/poster.png)
