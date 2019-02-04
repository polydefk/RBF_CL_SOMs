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

    def fit(self, data):
        for epoch in range(self.n_epochs):
            for idx in range(data.shape[0]):
                point = data[idx, :]
                winner_idx = self.find_minimum_distance(point)  # chicken dinner
                self.find_neighborhood(epoch)
    def get_distance(self, x, y):
        sub = x - y
        dist = np.dot(sub.T, sub)  # we skip the square because we only care about the index of the distance
        # print(dist)
        return dist

    def find_minimum_distance(self, point):
        # print(self.weights.shape)

        min_dist = self.get_distance(self.weights[0, :], point)
        min_index = 0
        for i in range(self.weights.shape[0]):
            dist = self.get_distance(self.weights[i, :], point)
            if dist < min_dist:
                min_dist = dist
                min_index = i

        return min_index

    def find_neighborhood(self,epoch):
        pass

    def update_weights(self):
        pass


if __name__ == "__main__":
    props, names, shape, epochs, eta = load_animals()
    som = SOM(shape=shape, n_epochs=epochs, eta=eta)
    som.fit(props)
