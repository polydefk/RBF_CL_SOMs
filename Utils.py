import numpy as np
from sklearn.metrics import mean_squared_error, zero_one_loss
import copy
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

np.random.seed(123)



def kernel(x, mean, std):
    diff = -(np.linalg.norm(x - mean)) ** 2

    standarized = diff / (2 * std ** 2)

    return np.exp(standarized)


def compute_rbf_centers(count, data=None):


    means = np.arange(0, 2 * np.pi, (2 * np.pi) / count)
    means = np.reshape(np.array(means), (len(means), 1))

    if data is not None:
        means = []
        ind = np.arange(0, data.shape[0])
        for i in range(count):
            index = np.random.choice(ind)
            means.append(data[index, :])

        means = np.array(means)

    return means


def compute_rbf_centers_competitive_learning(data, num_centers, eta, iterations, threshold = 0):
    dimensions = np.shape(data)[1]

    if dimensions == 1:
        rbf_centers = np.linspace(0, 2 * np.pi, num_centers)
        rbf_centers = np.reshape(rbf_centers, (rbf_centers.shape[0],1))
    else:
        rbf_centers = []
        ind = np.arange(0, data.shape[0])
        for i in range(num_centers):
            rbf_centers.append(data[np.random.choice(ind), :])
        rbf_centers = np.reshape(rbf_centers, (num_centers,2))


    for j in range(iterations):
        random_datapoint = data[np.random.choice(data.shape[0]),:]

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


def load_animals():
    with open('data/animals.dat') as file:
        lines = file.readlines()
        props = [line.split(',') for line in lines]
        props = [int(i) for i in props[0]]
        props = np.array(props)
        props = props.reshape((32, 84))

    with open("data/animalnames.txt") as f:
        lines = f.readlines()
        names = [line.strip('\t\n') for line in lines]
        names = np.array(names)


    return props, names

def load_cities():
    cities_data = np.loadtxt('data/cities.dat', delimiter=',', dtype=float)
    cities_labels = np.arange(0,len(cities_data),1)

    return cities_data, cities_labels

def load_MPs():
    votes = np.loadtxt('data/votes.dat', delimiter=',', dtype=float)
    mpnames = np.genfromtxt('data/mpnames.txt', delimiter='\n', dtype=str)
    mpsex = np.genfromtxt('data/mpsex.dat', delimiter=',', dtype=float)
    mpdistrict = np.genfromtxt('data/mpdistrict.dat', delimiter=',', dtype=float)
    mpparty = np.genfromtxt('data/mpparty.dat', delimiter=',', dtype=float)
    votes = np.reshape(votes, (349, 31))
    votes_labels = list(range(len(votes)))
    return votes, mpnames, mpsex, mpdistrict, mpparty, votes_labels



def load_ballist_data():
    data = []
    with open('data_lab2/ballist.dat', 'r') as f:
        next = f.readline()
        while next != "":
            list = next.replace('\t',' ').replace('\n', '').split(' ')
            list = [float(i) for i in list]
            data.append(list)
            next = f.readline()

    train = np.copy(data)
    train_data = train[:,0:2]
    train_labels = train[:,2:4]

    data = []
    with open('data_lab2/balltest.dat', 'r') as f:
        next = f.readline()
        while next != "":
            list = next.replace('\t',' ').replace('\n', '').split(' ')
            list = [float(i) for i in list]
            data.append(list)
            next = f.readline()

    test = np.copy(data)
    test_data = test[:,0:2]
    test_labels = test[:,2:4]

    return train_data, train_labels, test_data, test_labels



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
    plt.plot(epochs, pred, color='r')
    plt.plot(epochs, y_test, color='b')
    plt.xlabel('time')
    plt.ylabel('Absolut residual error')
    plt.legend(["Prediction", "","Actual"])
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

def plot_data_means(train_data, mean, mean_cl, title):

    plt.figure()
    plt.grid(True)

    train_data = plt.scatter(train_data[:, 0], train_data[:, 1], c='g', label="train data")
    nds = plt.scatter(mean[:, 0], mean[:, 1], c='b', label="No CL means")
    nds_cl = plt.scatter(mean_cl[:, 0], mean_cl[:, 1], c='r', label="CL means")

    plt.title(title)

    plt.ylim(0,1.5)

    plt.legend(handles=[train_data, nds, nds_cl],loc='upper left')
    plt.show()

def plot_data_means_1D(train_data, mean, mean_cl, title):
    plt.figure()
    plt.grid(True)

    train_data = plt.scatter(train_data, np.zeros(train_data.shape), c='g', label="train data")
    nds = plt.scatter(mean, np.zeros(mean.shape), c='b', label="No CL means", s=10)
    nds_cl = plt.scatter(mean_cl, np.zeros(mean_cl.shape), c='r', label="CL means", s = 10)

    plt.title(title)


    plt.legend(handles=[train_data, nds, nds_cl], loc='upper left')
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


def plot_cities_tour(cities_data, cities_prediction):
    verts = np.empty(cities_data.shape)
    i = 0
    for index in cities_prediction[:, 1]:
        verts[i] = cities_data[index, :]
        i += 1

    codes = [Path.LINETO] * len(cities_data)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    path = Path(verts, codes)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, lw=2)
    ax.add_patch(patch)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()

