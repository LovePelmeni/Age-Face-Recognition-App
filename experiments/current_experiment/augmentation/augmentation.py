from torchvision import transforms
from datasets import datasets
from current_experiment import constants

def apply_augmentations(dataset: datasets.FaceRecognitionDataset):
    """
    Function applies augmentations to a given dataset
    of images
    """
    transformations = transforms.Compose(
        [
            transforms.Resize((constants.INPUT_IMAGE_HEIGHT, constants.INPUT_IMAGE_WIDTH)),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage()
        ]
    )
    trans_dataset = transformations(dataset)
    return trans_dataset