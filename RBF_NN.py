import numpy as np
import Utils


class rbf_model(object):
    def __init__(self, data, targets, mean, cov, n_hidden_units, learning_rate):
        self.data = data
        self.targets = targets
        self.mean = mean
        self.cov = cov
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

    def update_weights(self,f_approx):
        self.weights += self.learning_rate * (self.targets - f_approx) * self.K

    def fit(self):
        f_approx = self.forward_pass()

        self.update_weights(f_approx)


    def forward_pass(self):
        f_approx = np.dot(self.weights, self.K)
        return f_approx


    def backwards_pass(self):

        pass

    def evaluate(self, targets,error_type):
        predictions = self.forward_pass()

        _, mse = Utils.compute_error(targets, predictions,error_type)
        return mse




if __name__ == "__main__":
    input_type = 'sin'
    error_type = 'mse'
    n_hidden_units = 8
    lr = 0.001
    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    mean = np.zeros(n_hidden_units)
    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    cov = np.ones(n_hidden_units)

    input = Utils.create_input(input_type)

    targets = []

    model = rbf_model(input, targets, mean, cov, n_hidden_units,lr)
    model.fit()
