import pytest 
import torch
from experiments.current_experiment import models, class_enums
import pandas
from PIL import Image

@pytest.fixture(scope='module')
def rest_form():
    return pandas.DataFrame(
        {
        '': [class_enums.ADULT, class_enums.ELDER, class_enums.CHILD, class_enums.TEENAGER],
        'images': [Image.open(""), Image.open(""), Image.open("")]
        }
    )

@pytest.fixture(scope='module')
def model():
    return torch.load(
        f='experiments/current_experiment/models'
    )

def test_model_prediction(rest_form):
    model = 