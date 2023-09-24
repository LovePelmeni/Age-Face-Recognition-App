import pytest
import torch
from experiments.current_experiment.models import models
import pandas
from PIL import Image
import torch.jit


@pytest.fixture(scope='module')
def rest_form():
    return pandas.DataFrame(
        {
            'class': [0, 1, 2, 2],
            'images': [Image.open(""), Image.open(""), Image.open("")]
        }
    )


@pytest.fixture(scope='module')
def model():
    return models.FaceRecognitionNet(
        trained_model=torch.jit.load(
            f='experiments/current_experiment/models'),
        loss_function=torch.nn.CrossEntropyLoss(),
        num_classes=3,
        learning_rate=3e-4
    )


def test_model_prediction(rest_form, model):
    predictions = model.forward(rest_form['images'])
    assert len(predictions) == len(rest_form['images'])
    assert all([str(pred).isdigit() for pred in predictions])
