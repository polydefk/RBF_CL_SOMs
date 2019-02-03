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

        else:
            for data_index in range(len(self.data)):
                inner = f_approx[data_index] - np.dot(self.transformed_data[data_index, :], self.weights)
                self.weights = np.add(self.weights, np.reshape(self.learning_rate * inner *
                                                               np.transpose(self.transformed_data[data_index, :]),
                                                               np.shape(self.weights)))

    def fit(self):

        train_error = 0
        for i in range(self.n_epochs):
            # we shuffle and then calculate kernels again
            [self.data, self.targets, self.transformed_data] = shuffle(self.data, self.targets, self.transformed_data)

            f_approx = self.forward_pass(self.transformed_data)

            self.update_weights(f_approx)

            train_error = self.evaluate(self.transformed_data, self.targets)

            if train_error < self.threshold:
                break

            if self.batch_train:
                print("number of hidden units: {0} train Error: {1}".format(self.n_hidden_units, train_error))
                break

        print("Epoch: {0} and Error: {1}".format(self.n_epochs, train_error))
        return self.weights

    def forward_pass(self, data, transform=False):
        if transform:
            data = self.calculate_transformed_data(data)
        f_approx = np.dot(self.weights.T, data.T)
        return f_approx.T

    def evaluate(self, data, targets, transform=False):

        predictions = self.forward_pass(data, transform)
        _, error = Utils.compute_error(predictions, targets)
        return error


if __name__ == "__main__":
    input_type = 'sin'
    np.random.seed(123)
    n_hidden_units = 4
    lr = 0.01
    n_epochs = 100
    batch_train = False

    cov = 0.5

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type)

    mean = Utils.compute_rbf_centers(n_hidden_units)
    model = rbf_model(x_train, y_train, mean, cov, n_epochs, n_hidden_units, lr, batch_train=batch_train)
    model.fit()
