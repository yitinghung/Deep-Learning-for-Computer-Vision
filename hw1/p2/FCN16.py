import torch
import torch.nn as nn
from func import set_parameter_requires_grad, write
from FCN32 import FCN32


class FCN16(nn.Module):
    def __init__(self, num_classes, feature_extract, path_log=None) -> None:
        super(FCN16,self).__init__()
        
        # load pre-trained FCN32 model
        pretrained_model = FCN32(num_classes=num_classes, feature_extract=feature_extract)
        
        if path_log:
            checkpoint = torch.load(path_log, map_location='cpu')
            write('load model:' 'from pre-trained FCN32 model')
            pretrained_model.load_state_dict(checkpoint['state_dict'])
        
        
        self.features4x = pretrained_model.features4x
        self.features8x = pretrained_model.features8x
        self.features16x = pretrained_model.features16x
        self.features32x = pretrained_model.features32x
        
        self.features32x_score = pretrained_model.features32x_score
        
        self.features32x_interp = nn.ConvTranspose2d(num_classes,num_classes,4,2)
        
        self.features16x_score = nn.Sequential(
                nn.Conv2d(512,2048,kernel_size=7,padding=3),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(2048,num_classes,kernel_size=1,padding=0), 
            )
        
        self.features16x_interp = nn.Sequential(nn.Conv2d(num_classes,num_classes,kernel_size=2,padding=0), 
                nn.ConvTranspose2d(num_classes,num_classes,32,16))
        
        if path_log:
            set_parameter_requires_grad(self.features4x, feature_extract)
            set_parameter_requires_grad(self.features8x, feature_extract)
            set_parameter_requires_grad(self.features16x, feature_extract)
            set_parameter_requires_grad(self.features32x, feature_extract)
            set_parameter_requires_grad(self.features32x_score, feature_extract)
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f4x = self.features4x(x)
        f8x = self.features8x(f4x)
        f16x = self.features16x(f8x)
        f32x = self.features32x(f16x)
        f32x_score = self.features32x_score(f32x)
        f32x_interp = self.features32x_interp(f32x_score)
        
        f16x_score = self.features16x_score(f16x)
    
        
        f16x_score = f16x_score + f32x_interp
        f16x_interp = self.features16x_interp(f16x_score)

        return f16x_interp
    
    
    