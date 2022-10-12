# TODO: create shell script for running your visualization code

if [ -e "p2/model.pth" ]; then
    # 目錄 /path/to/dir 存在
    echo "Directory CheckPoints exists."
else
    # 目錄 /path/to/dir 不存在
    echo "Directory CheckPoints does not exists."
    wget 'https://www.dropbox.com/s/82tp03qdqwmzws3/p2_model.pth?dl=0' -O p2/model.pth
    # Unzip the downloaded zip file
    #unzip ./CheckPoints.zip

    # Remove the downloaded zip file
    #rm ./CheckPoints.zip
fi

# Example
python3 ./p2/predict.py -i $1 -o $2
