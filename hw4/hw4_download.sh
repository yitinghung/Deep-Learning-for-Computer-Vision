# Download p1 model
if [ -e "p1/model.pth" ]; then
    echo "Directory CheckPoints exists."
else
    echo "Directory CheckPoints does not exists."
    wget 'https://www.dropbox.com/s/2y7alolbw6aol4c/max-acc.pth?dl=0' -O p1/model.pth
fi

# Download p2 model
if [ -e "p2/model.pth" ]; then
    echo "Directory CheckPoints exists."
else
    echo "Directory CheckPoints does not exists."
    wget 'https://www.dropbox.com/s/ysr9a0d2wfljwaq/C_finetune_ep340_0.4729.pth?dl=0' -O p2/model.pth
fi