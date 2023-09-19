from experiments.current_experiment import constants
from torchvision import transforms
from torchvision.transforms import v2

def apply_augmentations(dataset):
    """
    Function applies augmentations to a given dataset
    of images
    """
    transformations = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((constants.INPUT_IMAGE_HEIGHT, constants.INPUT_IMAGE_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            v2.RandomAdjustSharpness(sharpness_factor=constants.SHARPNESS_FACTOR, p=0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage()
        ]
    )
    trans_dataset = transformations(dataset)
    return trans_dataset

