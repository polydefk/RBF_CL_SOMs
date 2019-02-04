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


def compute_rbf_centers_competitive_learning(data, num_centers, eta, iterations, threshold = 0):

    rbf_centers = np.linspace(0, 2 * np.pi, num_centers)
    rbf_centers = np.reshape(rbf_centers, (rbf_centers.shape[0],1))
    data = np.reshape(data, (data.shape[0],))

    for j in range(iterations):
        random_datapoint = np.random.choice(data)

        # euclidean distance
        distances = []
        for center in rbf_centers:
            distances.append(np.linalg.norm(center - random_datapoint))
        distances = np.array(distances)

        if threshold == 0:
            indices = distances.argmin()
        else:
            values = np.abs(distances-np.min(distances)).T
            indices = np.where(values < threshold)
        rbf_centers[indices] += eta * (random_datapoint - rbf_centers[indices])

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
        y_train[y_train >= 0] = 1
        y_test[y_test < 0] = -1


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


def plot_many_lines(error, y_test, legend_names, title):
    # fig config
    plt.figure()
    plt.grid(True)

    epochs = np.arange(0, len(y_test), 1)

    plt.ylim(-1.55,2)
    for i in range(len(error)):
        plt.plot(epochs, error[i][:])

    plt.plot(epochs, y_test)

    plt.xlabel('time')
    plt.ylabel('Absolut residual error')

    plt.title(title)
    plt.legend(legend_names, loc='upper left')

    plt.show()


def plot_error_nodes(error, nodes, legend_names, title):
    # fig config
    plt.figure()
    plt.grid(True)


    plt.ylim(0,1)
    plt.xlim(1, nodes)
    plt.plot(np.arange(0, nodes, 1), error)

    plt.xlabel('time')
    plt.ylabel('Absolut residual error')

    plt.title(title)
    plt.legend(legend_names, loc='upper right')

    plt.show()
