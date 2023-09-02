from experiments.current_experiment import constants
from torchvision import transforms
import cv2


def apply_grayscale(image):
    scaled_image = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    return scaled_image


def apply_noise_reduction_filters():
    pass


def apply_normalization(image):
    norm_image = cv2.normalize(image)
    return norm_image


def apply_thresholding(gray_image):
    _, binary_image = cv2.threshold(
        gray_image,
        128,
        255,
        cv2.THRESH_BINARY
    )
    return binary_image


def apply_augmentations(dataset):
    """
    Function applies augmentations to a given dataset
    of images
    """
    transformations = transforms.Compose(
        [
            transforms.Resize((constants.INPUT_IMAGE_HEIGHT,
                              constants.INPUT_IMAGE_WIDTH)),
            transforms.ColorJitter(brightness=0.5),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.ToPILImage()
        ]
    )
    trans_dataset = transformations(dataset)
    return trans_dataset
