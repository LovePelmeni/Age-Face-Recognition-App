from fastapi import UploadFile, File
from experiments.current_experiment.datasets import datasets
import logging
from PIL import Image
from experiments.current_experiment.models import models
import fastapi.responses as resp
import torch.onnx

Logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='logs/rest_controllers.log')

model = torch.onnx.load("experiments/current_experiment/prod_models/neural_net.onnx")

async def predict_person_category(self, person_photo: UploadFile = File(...)):
    try:
        file_content = await person_photo.file.read()
        image_file = Image.open(file_content)
        predicted_class_number = model.forward(image_file)
        return resp.Response(
            status_code=200,
            content={'person_class': predicted_class_number}
        )
    except Exception as err:
        Logger.error(err)
