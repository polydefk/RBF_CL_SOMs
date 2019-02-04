import numpy as np
from sklearn.utils import shuffle
import Utils

np.random.seed(1)


class rbf_model(object):
    def __init__(self, data, targets, mean, cov, n_epochs, n_hidden_units, learning_rate,
                 error_type='delta', batch_train=False):
        self.data = data
        self.targets = targets
        self.mean = mean
        self.cov = cov
        self.error_type = error_type
        self.n_epochs = n_epochs
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
                error_ = self.targets[data_index] - np.dot(self.weights.T, np.reshape(self.transformed_data[data_index, :],(self.transformed_data[data_index, :].shape[0],1)))
                self.weights += (self.learning_rate * error_ * self.transformed_data[data_index, :]).T


    def fit(self):

        train_error = 0
        for i in range(self.n_epochs):
            # we shuffle and then calculate kernels again
            # [self.data, self.targets, self.transformed_data] = shuffle(self.data, self.targets, self.transformed_data)

            f_approx = self.forward_pass(self.transformed_data)

            self.update_weights(f_approx)

            train_error = self.evaluate(self.transformed_data, self.targets)

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
    num_hidden_units = 20
    noise = 0.1
    lr = 0.2
    n_epochs = 300
    batch_train = False

    cov = 1
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise)

    mean = Utils.compute_rbf_centers(num_hidden_units)

    model = rbf_model(x_train, y_train, mean, cov, n_epochs, num_hidden_units, lr, batch_train=batch_train)
    model.fit()

    predictions = model.forward_pass(x_test, True)
    error = model.evaluate(x_test, y_test, transform=True)
    print(error)
