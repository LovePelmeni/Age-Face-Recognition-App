from torchvision import transforms 
from datasets import datasets

def apply_augmentations(dataset: datasets.FaceRecognitionDataset):
    """
    Function applies augmentations to a given dataset
    of images
    """
    transformations = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToTensor(),
        ]
    )
    trans_dataset = transformations(dataset)
    return trans_dataset