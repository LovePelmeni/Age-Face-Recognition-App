
from experiments.current_experiment.datasets import datasets
from torchvision import models
from torchvision.ops import nms
import typing 
from PIL import Image
from torch import optim
from torch.utils import data
import torch
import logging 

logger = logging.getLogger(__name__)
file_handler = logging.FileHandler("./face_detection.log")
logger.addHandler(file_handler)

class HumanFaceDetector(object):

    """
    Human Face Detection Neural Network
    based on YOLO (You Only Look Once) Network
    """

    def __init__(self, 
        loss_function, 
        batch_size: int, 
        learning_rate: float, 
        weight_decay: float,
        max_epochs: int,
        device=None
    ):
        self.loss_function = loss_function
        self.max_epochs = max_epochs

        self.device = torch.device('cpu') if device is None else device
        self.model = models.detection.fasterrcnn_resnet50_fpn().to(device=self.device)

        self.batch_size = batch_size
        self.optimizer = optim.Adam(
            params=self.model.parameters(),
            weight_decay=weight_decay,
            lr=learning_rate,
        )

    def _select_valid_boxes(self, boxes, scores):
        return boxes[nms(boxes, scores, iou_threshold=0.5)]
    
    def train(self, dataset: datasets.FaceRecognitionDataset):

        dataloader = data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.model.train(mode=True)
        losses = []

        for _ in range(self.max_epochs):
            epoch_losses = []

            for labels, images in dataloader:
                lbls, imgs = labels.to(self.device), images.to(self.device)

                outputs = self.model.forward(images=imgs, targets=lbls).detach()
                loss = self.loss_function(labels, outputs)
                epoch_losses.append(loss.item())

            losses.append(sum(epoch_losses) / len(epoch_losses))
        return sum(losses) / len(losses)

    
    def predict_bbx(self, images: typing.List[Image.Image]):
        """
        Function predicts coordinates of the bounding boxes 
        for each given image

        images: typing.List[Image.Image] - array of images
        """ 
        self.model.eval()
        images_info = {} 

        for idx, image in enumerate(images):
            bbox_info = self.model.forward(images=[image])
            images_info[idx] = self._select_valid_boxes(
                boxes=bbox_info['bboxes'],
                scores=bbox_info['scores']
            )
        return images_info
    
    def checkpoint(self, epoch, loss, file_path):
        try:
            torch.save(
                {
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }, f=file_path
            )
        except(FileNotFoundError, FileExistsError) as file_err:
            logger.debug("failed to save checkpoint, file problem: %s" % file_err)

    def export(self, path: str):
        self.model.export(f=path)