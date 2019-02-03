import numpy as np
from sklearn.utils import shuffle
import Utils

np.random.seed(1)


class rbf_model(object):
    def __init__(self, data, targets, mean, cov, n_epochs, n_hidden_units, learning_rate, threshold=0.1,
                 error_type='delta', batch_train=False):
        self.data = data
        self.targets = targets
        self.mean = mean
        self.cov = cov
        self.error_type = error_type
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate
        self.batch_train = batch_train

        self.weights = np.random.randn(self.n_hidden_units, 1)
        self.transformed_data = self.calculate_transformed_data(data)

    def calculate_transformed_data(self, data):
        N = len(data)
        n = self.n_hidden_units
        K = np.zeros((N, n))
        for i in range(N):
            for j in range(n):
                K[i, j] = Utils.kernel(data[i, :],
                                       self.mean[j, :], self.cov)

        return K

    def update_weights(self, f_approx):
        # if batch use least squares else delta rule
        if self.batch_train:
            self.weights = np.linalg.solve(np.dot(self.transformed_data.T, self.transformed_data),
                                           np.dot(self.transformed_data.T, self.targets))
            # self.weights = np.linalg.inv(np.dot(self.transformed_data.T, self.transformed_data)).dot(
            #     self.transformed_data.T).dot(self.targets)


        else:
            self.weights += (self.learning_rate * np.dot((self.targets - f_approx).T, self.transformed_data)).T

    def fit(self, x_test, y_test):

        error = self.threshold + 1
        counter = 0
        # for i in range(self.n_epochs):
        while error > self.threshold:
            # we shuffle and then calculate kernels again
            [self.data, self.targets, self.transformed_data] = shuffle(self.data, self.targets, self.transformed_data)

            f_approx = self.forward_pass(self.data)

            self.update_weights(f_approx)

            train_error = self.evaluate(self.data, self.targets)

            test_error = self.evaluate(x_test, y_test)

            counter += 1
            # if error < self.threshold:
            #     break

            if batch_train:
                print("number of hidden units: {0} and train Error: {1}".format(self.n_hidden_units, train_error))
                print("number of hidden units: {0} and test Error: {1}".format(self.n_hidden_units, test_error))
                break

            print("Epoch: {0} and Error: {1}".format(counter, train_error))
        return self.weights

    def forward_pass(self, data):
        self.transformed_data = self.calculate_transformed_data(data)
        f_approx = np.dot(self.weights.T, self.transformed_data.T)
        return f_approx.T

    def evaluate(self, data, targets):

        transformed = self.forward_pass(data)
        _, error = Utils.compute_error(targets, transformed)
        return error


if __name__ == "__main__":
    input_type = 'sin'
    np.random.seed(123)
    n_hidden_units = 4
    lr = 0.01
    n_epochs = 100
    batch_train = True

    cov = 0.1

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type)

    mean = Utils.even_rbf_center(n_hidden_units)

    model = rbf_model(x_train, y_train, mean, cov, n_epochs, n_hidden_units, lr, batch_train=batch_train)
    model.fit(x_test, y_test)

    for i in np.arange(1, len(x_train)-1, 1):
        # mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, 1, 100)

        mean = Utils.compute_rbf_centers(i)

        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)

        model.fit(x_test, y_test)

        model.evaluate(x_test, y_test)
