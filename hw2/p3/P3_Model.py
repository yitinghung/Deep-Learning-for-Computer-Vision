import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(                      # input: (batch, 3, 28, 28)
            nn.Conv2d(3, 64, kernel_size=5,padding=2),  # (batch, 64, 28, 28)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),                            # (batch, 64, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5,padding=2), # (batch, 50, 14, 14)
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.MaxPool2d(2),                            # (batch, 50, 7, 7)
            nn.ReLU(True),
        )
    
    def forward(self, x):
        x = x.expand(len(x), 3, 28, 28)
        return self.conv(x)

class LabelPredictor(nn.Module):
    def __init__(self):
        super(LabelPredictor, self).__init__()          
        self.classifier = nn.Sequential(                # input: (batch, 50, 7, 7)
            nn.Linear(50 * 7 * 7, 100),                 # (batch, 100)
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),                         # (batch, 10)
        )

    def forward(self, x):
        return self.classifier(x)

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim): 
        super(DomainClassifier, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, x):
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DANN(nn.Module):
    def __init__(self, device):
        super(DANN, self).__init__()
        self.device = device
        self.feature = FeatureExtractor()
        self.label = LabelPredictor()
        self.domain = DomainClassifier(input_dim=50*7*7, hidden_dim=100)

    def adversarial(self, x, source=True, alpha=1):
        criterion = nn.BCELoss()
        if source:  # source: 1
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:       # target: 0
            domain_label = torch.zeros(len(x)).long().to(self.device)
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.domain(x).view(len(x))  
        loss = criterion(domain_pred, domain_label.float())
        return loss         

    def forward(self, x, alpha=1, source=True):
        x = x.expand(len(x), 3, 28, 28)
        x = self.feature(x)
        x = x.view(-1, 50*7*7)
        label_pred = self.label(x)
        domain_loss = self.adversarial(x, source=True, alpha=1)
        return label_pred, domain_loss

if __name__ == '__main__':
    # inputs = torch.zeros(3, 28, 28)
    inputs = torch.zeros(128, 3, 28, 28)
    dann  = DANN(device ='cpu')
    class_output, domain_output = dann(inputs)
    print(class_output.size())
    print(domain_output.size())

