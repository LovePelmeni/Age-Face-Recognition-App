from re import I
import torch
from torch import nn
import os
import logging
from datasets import datasets
from torch import optim
from torchvision import models
from torch.utils import data
from datetime import datetime
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
                 main_device,
                 loss_function,
                 batch_size: int,
                 max_epochs: int,
                 num_classes: int,
                 weights,
                 weight_decay: float = 0.01,
                 learning_rate: float = 0.001,
                 optimizer: optim.Optimizer = optim.Adam,
                 trained_model=None,
                 ):

        self.main_device = main_device if main_device else torch.device('cpu')
        self.model = trained_model if trained_model else models.resnet50(
            weights=weights).to(self.main_device)

        self.model.layer3.parameters()

        self.model.fc = nn.Linear(
            in_features=self.model.fc.in_features,
            out_features=num_classes
        )

        self.optimizer = optimizer if optimizer else optim.Adam(
            params=self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.max_epochs = max_epochs
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.freezed_params = set()

    def freeze_layers(self, parameters_names: list):
        """
        Function freezes layers parameters 
        to perform transfer learning task 
        
        Args:
            - parameters_names - list of parameter names
        """
        if len(parameters_names) == 0: return 
        for layer in [
            self.model.layer1, 
            self.model.layer2, 
            self.model.layer3, 
        ]:
            for name, param in layer.named_parameters():
                if name in parameters_names:
                    param.requires_grad = False 
                    self.freezed_params.add(param)
    
    def unfreeze_layers(self):
        """
        Function unfreezes given set of layer parameters
        """
        if len(self.freezed_params):
            for param in self.freezed_params:
                param.required_grad = True
            self.freezed_params.clear()
    
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
        self.model.eval(mode=True)
        prediction_probs = self.model.forward(x=images).cpu()
        return prediction_probs

    def evaluate(self, image_dataset: datasets.FaceRecognitionDataset):
        """
        Function evaluates model using given image dataset
        """
        if not len(image_dataset):
            return 0.0
        try:
            loss_function = self.loss_function(weight=image_dataset.weights)
            losses = []
            loader = data.DataLoader(
                dataset=image_dataset,
                shuffle=True,
                batch_size=self.batch_size
            )
            for labels, images in loader:
                predictions = self.model.forward(images).cpu()
                loss = loss_function(torch.tensor(labels), predictions)
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
        self.model.train(mode=True)

        training_set = data.DataLoader(
            dataset=image_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        total_loss = []
        loss_function = self.loss_function(weight=image_dataset.weights)

        for epoch in range(self.max_epochs):
            epoch_loss = []

            for batch_labels, batch_images in tqdm(training_set):
                
                dev_imgs = batch_images.to(self.main_device)
                predicted_classes = self.model.forward(dev_imgs).cpu()

                loss = loss_function(predicted_classes, torch.tensor(batch_labels))
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

