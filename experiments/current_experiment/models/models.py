import torch
from torch import nn
import os
import logging
from datasets import datasets
from torch.utils.data import DataLoader
import constants
from torch import optim

logger = logging.getLogger(__name__)
file_handler = logging.getLogger(__name__)
logger.addHandler(file_handler)

class FaceRecognitionNet(nn.Module):
    """
    Implementation of the Neural Network
    developed for Face Recognition, based on ResNet50 Architecture
    """ 

    def __init__(self, loss_function, num_classes: int, learning_rate: float = 0.001):

        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            momentum=0.9,
        )
        self.loss_function = loss_function

        self.final_output_layer = nn.Sequential([
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(dim=num_classes)
        ])

    def forward(self, X_data):

        # Applying Convolutional and Pooling layers
        try:
            output = self.layer1(X_data)
            output = self.layer2(output)
            output = self.layer3(output)
            output = self.layer4(output)
            output = self.layer5(output)
        except Exception as err:
            logger.error(err)
            raise SystemExit("Convolution Error, check logs")

        flattened = output.reshape(output.size(0), -1)
        try:
            # Fully-Connected Layers application
            output = self.fc_layer(flattened)
            output = self.fc_layer1(output)
            probs = self.fc_layer2(output)
        except Exception as err:
            logger.error(err)
            raise SystemExit("FC Layer Error, check logs")

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

        # paralelling Neural Network training

        paral_model = nn.DataParallel(
            module=self, 
            device_ids=torch.tensor([torch.device('mps'), torch.device('cuda')]),
            output_device=torch.device('cpu')
        )

        for batch_images, batch_labels in training_set:
            self.optimizer.zero_grad()

            predicted_classes = paral_model.forward(batch_images)
            loss = self.loss_function(predicted_classes, batch_labels)

            loss.backward()
            self.optimizer.step()

    def export(self, model_name: str, model_path: str):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(model=self, f=name)


