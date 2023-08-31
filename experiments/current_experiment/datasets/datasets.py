from torch.utils import data
from augmentation import augmentation
from PIL.Image import Image
import typing
class FaceRecognitionDataset(data.Dataset):
    """
    Implementation of the dataset used for 
    Face Recognition
    """
    def __init__(self, images, labels):
        self.images: typing.List[Image] = images
        self.labels: typing.List[int] = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        selected_data = self.images[idx]
        selected_labels = self.labels[idx]
        augmented_data = augmentation.apply_augmentations(
            dataset=selected_data
        )
        return selected_labels, augmented_data
