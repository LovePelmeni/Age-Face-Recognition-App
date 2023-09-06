import torch
from torch import nn
import os
import logging
from datasets import datasets
from torch.utils.data import DataLoader
import constants
from torch import optim
from ultralytics import YOLO
from PIL import Image
from torchvision.ops.boxes import nms

logger = logging.getLogger(__name__)
file_handler = logging.getLogger(__name__)
logger.addHandler(file_handler)

class FaceRecognitionNet(nn.Module):
    """
    Implementation of the Neural Network
    developed for Face Recognition
    """

    def __init__(self, num_classes: int, learning_rate: float = 0.001):

        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )

        self.loss_function = nn.CrossEntropyLoss()

        self.layer1 = nn.Sequential([
            nn.Conv2d(in_channels=3, out_channels=96, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])

        self.layer2 = nn.Sequential([
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=5, stride=1, padding=2),
            nn.BatchForm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        ])

        self.layer3 = nn.Sequential([
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(),
        ])

        self.layer4 = nn.Sequential([
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU()
        ])

        self.layer5 = nn.Sequential([
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=3, stride=1, padding=1),
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
            nn.Softmax(dim=num_classes)
        ])

    def forward(self, X_data):

        # Applying Convolutional and Pooling layers

        output = self.layer1(X_data)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = self.layer5(output)

        flattened = output.reshape(output.size(0), -1)

        # Fully-Connected Layers application
        output = self.fc_layer(flattened)
        output = self.fc_layer1(output)
        probs = self.fc_layer2(output)

        # finding best predicted class
        max_prob_idx = torch.argmax(input=probs)
        predicted_class = max_prob_idx
        return predicted_class

    def train(self, image_dataset: datasets.FaceRecognitionDataset):
        """
        Function trains Neural Network model using given image dataset
        """
        training_set = DataLoader(
            dataset=image_dataset,
            batch_size=constants.BATCH_SIZE,
            shuffle=True
        )
        for batch_images, batch_labels in training_set:
            self.optimizer.zero_grad()

            predicted_classes = self.forward(batch_images)
            loss = self.loss_function(predicted_classes, batch_labels)

            loss.backward()
            self.optimizer.step()

    def export(self, model_name: str, model_path: str):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(model=self, f=name)

class HumanFaceDetector(nn.Module):
    """
    Detector of Human Face

    Parameters:
    -----------

    detector (YOLO) - object detection model 
    available_device (torch.device) - available device for running training and prediction tasks
    """
    def __init__(self):
        self.detector = YOLO(model='yolov8n.yaml')
        self.available_device = torch.device(device='cuda' if torch.cuda.is_available() else 'cpu')
        self.detector.to(self.available_device)

    def fit(self, images, image_size: tuple):

        if not image_size or not isinstance(image_size, tuple):
            raise ValueError("Invalid image size")
        
        if len(image_size) != 2 or image_size[0] == 0 or image_size[1] == 0:
            raise ValueError("Invalid image size format")

        if not len(images):
            return 

        self.detector.train(
            data=images,
            epochs=len(images) ** 0.5,
            imgsz=image_size
        )

    def predict(self, image: Image):
        """
        Function detects human face on a given image
        """
        predictions = self.detector(image)

        # performing post-proces ssing techniques
        filtered_predictions = self._perform_post_processing(
            boxes=predictions['bbox'],
            scores=predictions['confidence_scores'],
            classes=predictions['classes'],
            names=predictions['names']
        )
        return filtered_predictions

    def export(self):
        """
        Function exports detector model
        to given ONNX path
        """
        exported_path = self.detector.export(format='onnx')
        return exported_path