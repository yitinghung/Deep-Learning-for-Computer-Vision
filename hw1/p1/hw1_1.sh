if [ -e "p1_model.ckpt" ]; then
    # 目錄 /path/to/dir 存在
    echo "Directory CheckPoints exists."
else
    # 目錄 /path/to/dir 不存在
    echo "Directory CheckPoints does not exists."
    #wget 'https://www.dropbox.com/s/dzg2w17yqgmof8l/CheckPoints.zip?dl=1' -O CheckPoints.zip
    #wget 'https://www.dropbox.com/s/c2yq3hgt23ef9zy/p1_model.ckpt?dl=0' -O p1_model.ckpt
    wget 'https://www.dropbox.com/s/qo6osy21z4nlyko/p1_model.ckpt?dl=0' -O p1_model.ckpt
    # Unzip the downloaded zip file
    #unzip ./CheckPoints.zip

    # Remove the downloaded zip file
    #rm ./CheckPoints.zip
fi

python3 p1_test.py -i $1 -o $2

# hw2_1.sh ./output_test/hw2_1/testing_50 ./output_test/hw2_1/output
