import torch
from torch import nn
import os
import logging
from datasets import datasets
from torch import optim
from torchvision import models
from torch.utils import data
import typing
from tqdm import tqdm

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
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.001,
                 momentum: float = 0.9,
                 optimizer: optim.Optimizer = optim.Adam,
                 trained_model=None
                 ):
        self.model = trained_model if trained_model else models.resnet50(
            weights=weights)

        self.model.fc = nn.Linear(
            in_features=self.fc.in_features,
            out_features=num_classes
        )

        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _save_checkpoint(self, epoch: int, loss: float, file_path: str):
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
            f='experiments/current_experiment/train_checkpoints/checkpoint_%s_%s.pt' % (
                epoch, datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            )
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
            raise RuntimeError(
                'certain required parameters was not presented in the state file')

    def predict(self, images):
        """
        Function returns softmax probabilities 
        of the output for each given class

        X_data (list of PIL Image objects) - training dataset
        """
        prediction_probs = self.model.forward(x=images)
        return prediction_probs

    def evaluate(self, image_dataset: datasets.FaceRecognitionDataset):
        """
        Function evaluates model using given image dataset
        """
        if not len(image_dataset):
            return 0.0
        try:
            losses = []
            loader = data.DataLoader(
                dataset=image_dataset,
                shuffle=True,
                batch_size=self.batch_size
            )
            for labels, images in loader:
                predictions = self.model.forward(images)
                loss = self.loss_function(labels, predictions)
                losses.append(loss.item())
            return sum(losses) / len(losses)

        except(Exception) as err:
            logger.error(err)
            raise RuntimeError(
            "Failed to evaluate model, internal error")

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
            batch_size=self.batch_size,
            shuffle=True
        )
        # paralelling Neural Network training

        paral_model = nn.DataParallel(
            module=self.model,
            device_ids=torch.tensor(
                [torch.device('mps'), torch.device('cuda')]),
            output_device=torch.device('cpu')
        )

        total_loss = []

        for epoch in range(self.max_epochs):
            epoch_loss = []

            for batch_images, batch_labels in training_set:

                self.optimizer.zero_grad()
                predicted_classes = paral_model.forward(batch_images)

                loss = self.loss_function(predicted_classes, batch_labels)
                epoch_loss.append(loss.item())

                loss.backward()
                self.optimizer.step()

            # saving checkpoints of the training process
            self._save_checkpoint(
                loss=torch.mean(epoch_loss),
                epoch=epoch
            )

            total_loss.append(torch.mean(epoch_loss))
        mean_loss = torch.mean(total_loss)
        return mean_loss

    def export(self, model_name: str, model_path: str):
        name = os.path.join(model_path, model_name + ".onnx")
        torch.onnx.export(model=self, f=name)
