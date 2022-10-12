# DLCV Final Project ( Skull Fractracture Detection )

# How to run your code?
    git clone https://github.com/ultralytics/yolov5.git
     
    python3 inf_preprocessing.py --only_test --test_root <Raw TESTING DATA PATH> --out_test_root ./data/test
     
    python3 yolov5/detect.py --source ./data/test/ --img 512 --max-det 6 --weights ./ckpt/yolov5s_adamw.pt --conf-thres 0.5 --save-txt --project ./result --name adamw
    python3 yolov5/detect.py --source ./data/test/ --img 512 --max-det 6 --weights ./ckpt/yolov5s_sgd.pt --conf-thres 0.5 --save-txt --project ./result --name sgd
     
    python3 ensemble_inference.py --exp_path_1 ./result/adamw --exp_path_2 ./result/sgd --output_csv ./result.csv

    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/ntudlcv/DLCV-Fall-2021-Final-1-<team name>.git
  
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://drive.google.com/drive/folders/13PQuQv4dllmdlA7lJNiLDiZ7gOxge2oJ?usp=sharing) to view the slides of Final Project - Skull Fracture Detection. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `skull`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1i2MlS-eAkx0bFFKirSEmQyp5_FIPJO7p/view?fbclid=IwAR3-xGO3EOTQBoTR_PtCAlHIVK_QxMz-WmzoiZrSC8PWsdM1k0xGU5HW6vg) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `skull` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Evaluation Code
In the starter code of this repository, we have provided a python script for evaluating the results for this project. For Linux users, use the following command to evaluate the results.
```bash
python3 for_students_eval.py --pred_file <path to your prediction csv file> --gt_file <path to the ground-truth csv file>
```

# Submission Rules
### Deadline
110/1/18 (Tue.) 23:59 (GMT+8)
    
# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
