import numpy as np
from sklearn.utils import shuffle

import Utils

np.random.seed(123)



class SOM(object):
    def __init__(self, shape, n_epochs, eta):

        self.weights = np.random.random(size=shape)
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

    def predict(self, test, labels):
        predictions = []
        for index in range(test.shape[0]):
            point = test[index, :].copy()

            distance = self.get_distance_matrix(point)
            winner = np.argmin(distance)
            predictions.append([winner, labels[index]])
        predictions = np.array(predictions, dtype=object)
        predictions = predictions[predictions[:, 0].argsort()]
        return predictions


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


    props, names = Utils.load_animals()
    cities_data, cities_labels = Utils.load_cities()
    votes, mpnames, mpsex, mpdistrict, mpparty, votes, votes_labels = Utils.load_MPs()

    # props = shuffle(props)

    weight_shape = (100, 84)
    epochs = 20
    eta = 0.2


    som = SOM(shape=weight_shape, n_epochs=epochs, eta=eta)
    som.fit(props)

    pred = som.predict(props, names)

    print(pred)
