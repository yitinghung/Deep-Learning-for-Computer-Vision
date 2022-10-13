# TODO: create shell script for running your ViT testing code

if [ -e "p1/model.pth" ]; then
    # 目錄 /path/to/dir 存在
    echo "Directory CheckPoints exists."
else
    # 目錄 /path/to/dir 不存在
    echo "Directory CheckPoints does not exists."
    wget 'https://www.dropbox.com/s/48yjef90rrakhoo/p1_model.pth?dl=0' -O p1/model.pth
    # Unzip the downloaded zip file
    #unzip ./CheckPoints.zip

    # Remove the downloaded zip file
    #rm ./CheckPoints.zip
fi

# Example
python3 Test.py -i $1 -o $2
