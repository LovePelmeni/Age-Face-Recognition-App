import torch
from torch import nn
import os
import logging
from datasets import datasets
import constants
from torch import optim
from datetime import datetime
from torchvision import models
from torch.utils import data

logger = logging.getLogger(__name__)
file_handler = logging.getLogger(__name__)
logger.addHandler(file_handler)

class FaceRecognitionNet(object):
    """
    Implementation of the Neural Network
    developed for Face Recognition, based on ResNet50 Architecture
    """ 

    def __init__(self, 
        loss_function, 
        num_classes: int, 
        weights,
        main_device,
        weight_decay: float = 0.01,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
    ):
        self.model = models.resnet50(weights=weights)
        self.main_device = main_device
        self.loss_function = loss_function

        self.model.fc = nn.Linear(
            in_features=self.fc.in_features, 
            out_features=num_classes
        )

        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    def save_checkpoint(self, epoch: int, loss: float, file_path: str):
        """
        Function saves model intermediate training checkpoints 
        """
        torch.save(
            obj={
                "optimizer_state_dict": self.optimizer.state_dict(),
                "model_state_dict": self.module.state_dict(),
                "loss": loss,
                "epoch": epoch
            }, 
            f=file_path
        )
    def load_state(self, file_path: str):
        try:
            state = torch.load(f=file_path)
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.load_state_dict(state['model_state_dict'])

        except FileNotFoundError as err:
            logger.debug(err)
            raise RuntimeError("Failed to load state, file does not exist")
            
        except KeyError as err:
            logger.debug(err)
            raise RuntimeError('certain required parameters was not presented in the state file')

    def predict(self, X_data):
        """
        Function returns softmax probabilities 
        of the output for each given class

        X_data (list of PIL Image objects) - training dataset
        """
        prediction_probs = self.model.forward(x=X_data)
        return prediction_probs

    def train(self, image_dataset: datasets.FaceRecognitionDataset):
        """
        Function trains Neural Network model using given image dataset

        Args:
            image_dataset - dataset of images, used for training 
        """
        # turning model into training process
        super().train(mode=True)

        training_set = data.DataLoader(
            dataset=image_dataset,
            batch_size=constants.BATCH_SIZE,
            shuffle=True
        )
        # paralelling Neural Network training

        paral_model = nn.DataParallel(
            module=self.model,
            output_device=torch.device('cpu')
        )

        total_loss = []

        for epoch in range(self.max_epochs):
            epoch_loss = []

            for batch_labels, batch_images in training_set:
                
                self.optimizer.zero_grad()

                predicted_classes = paral_model.forward(
                batch_images.to(self.main_device)).detach()

                loss = self.loss_function(predicted_classes, batch_labels)
                epoch_loss.append(loss.item())

                # backpropagation
                loss.backward()
                self.optimizer.step()

            # saving checkpoints of the training process
            self.save_checkpoint(
                loss=torch.mean(epoch_loss),
                epoch=epoch
            )
            total_loss.append(torch.mean(epoch_loss))
        mean_loss = torch.mean(total_loss)
        return mean_loss

    def export(self, model_name: str, model_path: str):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(model=self, f=name)
