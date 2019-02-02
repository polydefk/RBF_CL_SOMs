import numpy as np
from sklearn.metrics import mean_squared_error, zero_one_loss


def kernel(x, mean, std):
    diff = -np.square(x - mean)

    standarized = diff / 2 * np.square(std)

    K = np.exp(standarized)

    return K


def create_dataset(type):
    step = 0.1

    x_train = np.arange(0, 2 * np.pi, step)
    y_train = np.sin(2 * x_train)

    x_test = np.arange(0.05, 2 * np.pi, step)
    y_test = np.sin(2 * x_test)

    y_train = np.reshape(y_train, (y_train.shape[0],1))
    y_test = np.reshape(y_test, (y_test.shape[0],1))

    if type == 'sin':
        return x_train, y_train, x_test, y_test

    elif type == 'square':
        y_train = [-1.0 if i >= 0.0 else 1.0 for i in y_train]
        y_test = [-1.0 if i >= 0.0 else 1.0 for i in y_test]
        return x_train, y_train, x_test, y_test


def compute_error(targets, predictions, error_type='are', binary=False):
    loss = 0
    error = 0
    if error_type is 'mse':
        error = mean_squared_error(targets, predictions)
    # fraction of misclassifications
    if error_type is 'are':
        error = np.mean(np.abs(targets - predictions))

    if binary:
        predictions = np.where(predictions >= 0, 1, -1)
        loss = zero_one_loss(targets, predictions, normalize=True)

    return loss * 100, error
