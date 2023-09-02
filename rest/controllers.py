from fastapi import UploadFile, File
import logging
from PIL import Image
import fastapi.responses as resp
import torch.onnx
import torch

Logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(filename='logs/rest_controllers.log')
Logger.addHandler(file_handler)

# Loading Neural Network..
try:
    model = torch.jit.load("experiments/current_experiment/prod_models/neural_net.onnx")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # entering evaluation mode for Neural Network
    model.eval()

except Exception as err:
    Logger.critical(err)
    raise SystemExit("Failed to load neural network, check 'rest_controllers' logs...")

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
