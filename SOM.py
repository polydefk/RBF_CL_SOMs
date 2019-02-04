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
        self.neighbors_num = 50
        self.eta = eta

    def fit(self, data):
        for epoch in range(self.n_epochs):
            for point in data:

                distance = self.get_distance_matrix(point)
                winner = np.argmin(distance)

                min_boundary = max(0, winner - self.neighbors_num)

                max_boundary = min(100, winner + self.neighbors_num)

                self.update_weights(point, min_boundary, max_boundary)
            self.update_params(epoch)

    def predict(self, test):
        indices = []
        for point in test:

            distance = self.get_distance_matrix(point)
            winner = np.argmin(distance)
            indices.append(winner)

        return np.array(indices)

    def get_distance(self, x, y):
        sub = x - y
        dist = np.dot(sub.T, sub)
        return dist

    def get_distance_matrix(self, point):
        distances = []
        for i in range(self.weights.shape[0]):
            dist = self.get_distance(self.weights[i, :], point)
            distances.append(dist)
        return np.array(distances)

    def update_weights(self, point, min_boundary, max_boundary):
        for j in range(min_boundary, max_boundary):
            self.weights[j] += self.eta * (point - self.weights[j])

    def update_params(self, epoch):
        self.neighbors_num = int(self.neighbors_num * (1 - epoch / self.n_epochs))



if __name__ == "__main__":
    props, names, shape, epochs, eta = load_animals()
    som = SOM(shape=shape, n_epochs=epochs, eta=eta)
    som.fit(props)

    pred_indices = som.predict(props)

    Z = [x for _, x in sorted(zip(pred_indices, names))]

    print(np.transpose(Z))
