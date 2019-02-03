import numpy as np
from sklearn.metrics import mean_squared_error, zero_one_loss
import copy
import matplotlib.pyplot as plt

def kernel(x, mean, std):
    diff = -(np.linalg.norm(x - mean)) ** 2

    standarized = diff / (2 * std ** 2)

    return np.exp(standarized)


def compute_rbf_centers(count):
    means = np.arange(0, 2 * np.pi, (2 * np.pi) / count)
    means = np.reshape(np.array(means), (len(means), 1))
    return means


def compute_rbf_centers_competitive_learning(data, num_centers, eta, iterations):
    np.random.shuffle(data)

    rbf_centers = copy.deepcopy(data[0:num_centers])  # can use mean of data

    for j in range(iterations):
        random_datapoint = data[np.random.randint(0, len(data)), :]

        distances = []
        for center in rbf_centers:
            distances = np.append(distances, (np.linalg.norm(center - random_datapoint)))

        closer_rbf_center = distances.argmin()
        rbf_centers[closer_rbf_center] += eta * (random_datapoint - rbf_centers[closer_rbf_center])
    return rbf_centers


def create_dataset(type, noise=0):
    step = 0.1

    x_train = np.arange(0, 2 * np.pi, step)
    y_train = np.sin(2 * x_train)

    x_test = np.arange(0.05, 2 * np.pi, step)
    y_test = np.sin(2 * x_test)

    x_train = np.reshape(x_train, (x_train.shape[0], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], 1))

    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))



    if type == 'sin':
        pass

    elif type == 'square':
        y_train = [-1.0 if i >= 0.0 else 1.0 for i in y_train]
        y_test = [-1.0 if i >= 0.0 else 1.0 for i in y_test]


    if noise > 0:
        x_train += np.random.normal(0, noise, x_train.shape)
        y_train += np.random.normal(0, noise, y_train.shape)
        x_test += np.random.normal(0, noise, x_test.shape)
        y_test += np.random.normal(0, noise, y_test.shape)

    return x_train, y_train, x_test, y_test

def even_rbf_center(count):
    mu_list = []
    for i in range(count):
        mu_list.append(i * 2 * np.pi / (count))

    mu_list = np.reshape(np.array(mu_list), (len(mu_list), 1))

    return mu_list


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



def plot_pred_actual(pred, y_test, title):
    # fig config
    plt.figure()
    plt.grid(True)
    epochs = np.arange(0, len(pred), 1)

    plt.title(title)
    plt.plot(epochs, pred, color='r', label="Prediction")
    plt.plot(epochs, y_test, color='b', label="Actual")
    plt.xlabel('time')
    plt.ylabel('Absolut residual error')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs,np.abs(pred-y_test),color='r',label = 'Difference')
    plt.xlabel('time')
    plt.ylabel('Absolute residual error')
    plt.legend()
    plt.show()
