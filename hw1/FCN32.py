import torch
import torch.nn as nn
from torchvision import models
from func import set_parameter_requires_grad


class FCN32(nn.Module):
    def __init__(self, num_classes, feature_extract) -> None:
        super(FCN32,self).__init__()
        
        pretrained_model = models.vgg16(pretrained=True)
        features = list(pretrained_model.features.children())
        
        self.features4x = nn.Sequential(*features[0:10])
        self.features8x = nn.Sequential(*features[10:17])
        self.features16x = nn.Sequential(*features[17:24])
        self.features32x = nn.Sequential(*features[24:31])
        
        self.features32x_score = nn.Sequential(
                nn.Conv2d(512,4096,kernel_size=7,padding=3),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(4096,4096,kernel_size=2,padding=0),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Conv2d(4096,num_classes,kernel_size=1,padding=0), 
            )
        
        self.features32x_interp = nn.ConvTranspose2d(num_classes,num_classes,64,32)
        
        
        set_parameter_requires_grad(self.features4x, feature_extract)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features4x = self.features4x(x)
        features8x = self.features8x(features4x)
        features16x = self.features16x(features8x)
        features32x = self.features32x(features16x)
        
        features32x_score = self.features32x_score(features32x)
        features32x_interp = self.features32x_interp(features32x_score)
        
        return features32x_interp
    
    
    
    