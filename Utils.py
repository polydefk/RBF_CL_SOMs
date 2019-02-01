import numpy as np
from sklearn.metrics import mean_squared_error, zero_one_loss


def kernel(x, mean, std):
    diff = -np.square(x - mean)

    standarized = diff / 2 * np.square(std)

    K = np.exp(standarized)

    return K


def create_input(type):
    step = 0.1

    time = np.arange(0, 2 * np.pi, step)
    x = np.sin(2 * time)

    if type == 'sin':

        return x

    elif type == 'square':
        x = [-1.0 if i >= 0.0 else 1.0 for i in x]
        return x


def compute_error(targets, predictions, error_type, binary=False):
    loss = 0
    mse = 0

    if error_type is 'mse':
        mse = mean_squared_error(targets, predictions)
    # fraction of misclassifications
    if binary:
        predictions = np.where(predictions >= 0, 1, -1)
        loss = zero_one_loss(targets, predictions, normalize=True)

    return loss * 100, mse
