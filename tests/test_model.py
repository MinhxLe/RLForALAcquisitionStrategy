import numpy as np
from src.model.mnist_model import MNISTModel


def test_MNISTModel_input():
    test_images = np.random.random((32, 28, 28, 1))
    model = MNISTModel()
    output = model(test_images)
    assert output.shape == (32, 10)


test_MNISTModel_input()
