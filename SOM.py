import numpy as np
from sklearn.utils import shuffle
from scipy.spatial.distance import cityblock as manhattan
import Utils

np.random.seed(123)


class SOM(object):
    def __init__(self, shape, n_epochs, eta, neighbors_num, neighbohood_function=None):

        self.weights = np.random.random(size=shape)
        self.n_epochs = n_epochs
        self.neighbors_num = neighbors_num
        self.eta = eta
        self.neighbohood_function = neighbohood_function

    def fit(self, data):
        for epoch in range(self.n_epochs):
            if epoch % 5 == 0:
                print(epoch)

            for point in data:
                neighbors = self.find_neighbohood(point)

                self.update_weights(point, neighbors)
            self.update_params(epoch)

    def predict(self, test, labels):
        predictions = []
        for index in range(test.shape[0]):
            point = test[index, :].copy()

            winner = self.get_winner(point)

            predictions.append([winner, labels[index]])

        predictions = np.array(predictions, dtype=object)
        predictions = predictions[predictions[:, 0].argsort()]
        return predictions

    def predict_2_D(self, test):
        predictions = []
        for index in range(test.shape[0]):
            point = test[index, :].copy()

            winner = self.get_winner(point)

            predictions.append((int(winner / 10), winner % 10) )

        predictions = np.array(predictions)
        # predictions = predictions[predictions[:, 0].argsort()]
        return predictions

    def find_neighbohood(self, point):
        winner = self.get_winner(point)
        neighbors = []

        if self.neighbohood_function is None:
            diff = int(self.neighbors_num / 2)
            min_boundary = max(0, winner - diff)
            max_boundary = min(100, winner + diff)
            neighbors = range(min_boundary, max_boundary)

        elif self.neighbohood_function == 'circular':
            neighbors = self.find_circular_neighbourhood(winner)
        elif self.neighbohood_function == 'manhattan':
            neighbors = self.find_manhattan_neighbourhood(winner)
        return neighbors

    def get_manhattan_distance(self, x, y):
        dist = manhattan(x, y)
        return dist

    def get_distance(self, x, y):
        sub = x - y
        dist = np.dot(sub.T, sub)
        return dist

    def get_winner(self, point):
        distances = []
        for i in range(self.weights.shape[0]):
            dist = self.get_distance(self.weights[i, :], point)
            distances.append(dist)

        winner = np.argmin(np.array(distances))
        return winner

    def find_circular_neighbourhood(self, winner):
        diff = int(self.neighbors_num / 2)
        weight_length = self.weights.shape[0]

        lower = (winner - diff) % weight_length
        upper = (winner + diff) % weight_length

        return range(lower, upper)

    def update_weights(self, point, neighbors):
        for j in neighbors:
            self.weights[j] += self.eta * (point - self.weights[j])

    def update_params(self, epoch):
        self.neighbors_num = int(self.neighbors_num * (1 - epoch / self.n_epochs))

    def find_manhattan_neighbourhood(self, winner):
        winner_point = (int(winner / 10), winner % 10)

        neighbours = []
        for i in range(self.weights.shape[0]):
            point = (int(i / 10), i % 10)

            distance = manhattan(point, winner_point)

            if distance > self.neighbors_num:
                continue

            neighbours.append(i)

        return neighbours

if __name__ == "__main__":
    cities_data, cities_labels = Utils.load_cities()
    votes, mpnames, mpsex, mpdistrict, mpparty, votes_labels = Utils.load_MPs()

    # props = shuffle(props)

    weight_shape = (10, 2)
    epochs = 20
    eta = 0.2

    som = SOM(shape=weight_shape, n_epochs=epochs, eta=eta, neighbors_num=2, neighbohood_function='manhattan')
    som.fit(cities_data)

    pred = som.predict(cities_data, cities_labels)

    print(pred)
    Utils.plot_cities_tour(cities_data, pred)
