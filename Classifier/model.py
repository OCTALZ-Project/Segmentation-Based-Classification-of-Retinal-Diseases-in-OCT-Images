import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50

# Define the ResNet classifier with adjustable dropout rates
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.5):
        super(ResNetClassifier, self).__init__()
        self.model = resnet50(pretrained=True)
        
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes, bias=True)


    def forward(self, x):
        return self.model(x)


    def forward(self, x):
        return self.model(x)

