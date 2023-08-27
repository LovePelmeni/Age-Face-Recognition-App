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
    def __init__(self):
        pass

    def train(self, X_data):
        pass 
    
    def forward(self, X_data):
        pass
    
    @staticmethod
    def export(
        model,
        model_name: str, 
        model_path: str
    ):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(
            model=model,
            f=name
        )