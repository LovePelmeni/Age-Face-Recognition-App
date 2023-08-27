from torch.utils import data 
import torch

class FaceRecognitionDataset(data.Dataset):
    """
    Implementation of the dataset used for 
    Face Recognition
    """
    def __init__(self, images, labels):
        self.images = images 
        self.labels = labels 

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        pass
