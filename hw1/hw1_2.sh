if [ -e "p2_model.ckpt" ]; then
    # 目錄 /path/to/dir 存在
    echo "Directory CheckPoints exists."
else
    # 目錄 /path/to/dir 不存在
    echo "Directory CheckPoints does not exists."
    #wget 'https://www.dropbox.com/s/qo6osy21z4nlyko/p1_model.ckpt?dl=0' -O p1_model.ckpt
    wget 'https://www.dropbox.com/s/ztcn3n3tkrdwbno/p2_model.ckpt?dl=0' -O p2_model.ckpt
    # Unzip the downloaded zip file
    #unzip ./CheckPoints.zip

    # Remove the downloaded zip file
    #rm ./CheckPoints.zip
fi

python3 p2_test.py -i $1 -o $2

# hw2_1.sh ./output_test/hw2_1/testing_50 ./output_test/hw2_1/output
