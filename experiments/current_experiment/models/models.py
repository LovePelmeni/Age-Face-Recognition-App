import torch 
from torch import nn
import os 
import logging 

logger = logging.getLogger(__name__)
file_handler = logging.getLogger(__name__)
logger.addHandler(file_handler)

class FaceRecognitionNet(nn.Module):
    """
    Implementation of the Neural Network
    developed for Face Recognition
    """
    def __init__(self, num_classes: int):

        self.layer1 = nn.Sequential([
            nn.Conv2d(in_channels=3, out_channels=96, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])

        self.layer2 = nn.Sequential([
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchForm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])

        self.layer3 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),
        ])

        self.layer4 = nn.Sequential([
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU()
        ])
        
        self.layer5 = nn.Sequential([
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])
        
        self.fc_layer = nn.Sequential([
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096),
            nn.ReLU()
        ])

        self.fc_layer1 = nn.Sequantial([
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU()
        ])

        self.fc_layer2 = nn.Sequential([
            nn.Linear(in_features=4096, out_features=num_classes),
        ])

    def forward(self, X_data):

        output = self.layer1(X_data)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)

        output = output.reshape(output.size(0), -1)
        output = self.fc_layer(output)
        output = self.fc_layer1(output)
        output = self.fc_layer2(output)
        return output
    
    @staticmethod
    def export(model, model_name: str, model_path: str):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(model=model, f=name)
