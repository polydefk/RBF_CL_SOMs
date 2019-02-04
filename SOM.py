import numpy as np

np.random.seed(1234)


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

    weight_shape = (100, 84)
    epochs = 20
    eta = 0.2
    return props, names, weight_shape, epochs, eta


class SOM(object):
    def __init__(self, shape, n_epochs, eta):

        self.weights = np.random.normal(size=shape)
        self.n_epochs = n_epochs
        self.eta = eta
        self.neighbors = None
        self.neighbors_num = 50
        self.a = 0.5

    def fit(self, data):
        for epoch in range(self.n_epochs):
            for idx in range(data.shape[0]):
                point = data[idx, :]

                sorted_dist, sorted_ind = self.find_sorted_distances_indices(point)

                self.neighbors = np.zeros(self.weights.shape[0])

                self.neighbors[sorted_ind[0:self.neighbors_num]] = 1

                self.update_weights(point)

            self.update_params(epoch)

    def get_distance(self, x, y):
        sub = x - y
        dist = np.dot(sub.T, sub)  # we skip the square because we only care about the index of the distance
        # print(dist)
        return dist

    def find_sorted_distances_indices(self, point):
        distances = []
        for i in range(self.weights.shape[0]):
            dist = self.get_distance(self.weights[i, :], point)
            distances.append(dist)

        indices = sorted(range(len(distances)), key=distances.__getitem__)
        sorted_dist = sorted(distances)

        return sorted_dist, indices

    def update_weights(self, point):

        for i in range(self.weights.shape[0]):
            dist = self.get_distance(self.weights[i, :], point)

            self.weights[i, :] += self.eta * self.neighbors * dist

    def update_params(self, epoch):
        self.eta = self.a * (self.eta ** (epoch / self.n_epochs))
        self.neighbors_num = self.a * (self.neighbors_num ** (epoch / self.n_epochs))


if __name__ == "__main__":
    props, names, shape, epochs, eta = load_animals()
    som = SOM(shape=shape, n_epochs=epochs, eta=eta)
    som.fit(props)
