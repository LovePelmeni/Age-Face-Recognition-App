import pytest 
from PIL import Image 

@pytest.fixture(scope='module')
def valid_image():
    return Image.open()

@pytest.fixture(scope='module')
def invalid_image():
    return Image.open()

def test_prediction_endpoint():
    pass

def test_fail_prediction_endpoint():
    pass