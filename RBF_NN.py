import numpy as np


def kernel(x, mean, std):
    diff = -np.square(x - mean)

    standarized = diff / 2 * np.square(std)

    K = np.exp(standarized)

    return K


def create_input(type):
    step = 0.1

    time = np.arange(0, 2 * np.pi, step)
    x = np.sin(2 * time)

    if type == 'sin':
        return x

    elif type == 'square':
        x = [-1.0 if i >= 0.0 else 1.0 for i in x]
        return x


class rbf_model(object):
    def __init__(self, data, mean, std, n_hidden_units):
        self.data = data
        self.mean = mean
        self.cov = cov
        self.n_hidden_units = n_hidden_units
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
            K.append(kernel(self.data, self.mean[i], self.cov[i]))
        return np.array(K)

    def forward_pass(self):
        f_approx = np.dot(self.weights, self.K)
        pass # END here
    
    def fit(self):
        self.forward_pass()


if __name__ == "__main__":
    type = 'sin'

    n_hidden_units = 8

    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    mean = np.zeros(n_hidden_units)
    # NEED TO FIND A WAY TO COMPUTE THIS MEAN AND COV
    cov = np.ones(n_hidden_units)

    input = create_input(type)

    model = rbf_model(input, mean, cov, n_hidden_units)
    model.fit()
