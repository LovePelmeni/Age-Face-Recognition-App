from PIL.Image import Image
import typing
from torch.utils import data
from torchvision import transforms

class FaceRecognitionDataset(data.Dataset):
    """
    Implementation of the dataset used for 
    Face Recognition
    """

    def __init__(self, images, labels, transformations=None):
        self.images: typing.List[Image] = images
        self.labels: typing.List[int] = labels
        self.transformations = transforms.Compose(transformations) if transformations else None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        selected_data = self.images[idx]
        selected_labels = self.labels[idx]
        if self.transformations is not None:
            selected_data = self.transformations(selected_data)
        return selected_labels, selected_data
