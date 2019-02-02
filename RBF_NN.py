import numpy as np
import Utils


class rbf_model(object):
    def __init__(self, data, targets, mean, cov, n_epochs, n_hidden_units, learning_rate, threshold=0.1,
                 error_type='delta'):
        self.data = data
        self.targets = targets
        self.mean = mean
        self.cov = cov
        self.error_type = error_type
        self.n_epochs = n_epochs
        self.threshold = threshold
        self.n_hidden_units = n_hidden_units
        self.learning_rate = learning_rate
        self.weights = self.initialize_weights()
        self.K = self.initialize_kernels()

    def initialize_weights(self):
        n_hidden_units = self.n_hidden_units
        min = -1 / np.sqrt(n_hidden_units)
        max = 1 / np.math.sqrt(n_hidden_units)

        return np.random.normal(min, max, size=(1, n_hidden_units))

    def initialize_kernels(self):
        K = []
        for i in range(self.n_hidden_units):
            K.append(Utils.kernel(self.data, self.mean[i], self.cov[i]))
        return np.array(K)

    def update_weights(self, f_approx):
        if self.error_type is 'delta':
            self.weights += self.learning_rate * np.dot((self.targets - f_approx).T, self.K.T)
        elif self.error_type is 'least_square':
            pass

    def fit(self):

        error = 1
        counter = 0
        # for i in range(self.n_epochs):
        while error > self.threshold:

            f_approx = self.forward_pass()

            self.update_weights(f_approx)

            error = self.evaluate(f_approx, self.targets)
            print("Epoch: {0} and Error: {1}".format(counter, error))
            counter += 1
            # if error < self.threshold:
            #     break

        return self.weights

    def forward_pass(self):
        f_approx = np.dot(self.weights, self.K)
        return f_approx.T

    def backwards_pass(self):
        pass

    def evaluate(self, predictions, targets):

        _, error = Utils.compute_error(targets, predictions)
        return error


if __name__ == "__main__":
    input_type = 'sin'
    error_type = 'mse'
    n_hidden_units = 10
    lr = 0.01
    n_epochs = 100
    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    mu, sigma = 0, 0.1  # mean and standard deviation

    mean = np.random.normal(mu, sigma, n_hidden_units)
    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    cov = np.random.normal(mu, sigma, n_hidden_units)

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type)

    targets = []

    model = rbf_model(x_train, y_train, mean, cov, n_epochs, n_hidden_units, lr)
    model.fit()
