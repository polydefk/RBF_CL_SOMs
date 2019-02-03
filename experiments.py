import Utils
import numpy as np
from RBF_NN import rbf_model


def experiment_rbf_with_noise():
    input_type = 'sin'
    np.random.seed(123)
    noise = 0.1
    lr = 0.01
    n_epochs = 100
    batch_train = False

    cov = 0.5

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    predictions = []
    best_error = 100
    best_node = -1
    for i in np.arange(1, len(x_train), 1):
        # mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, eta=0.2, iterations=100)

        mean = Utils.compute_rbf_centers(i)

        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)

        model.fit()

        predictions = model.forward_pass(x_test, True)
        error = model.evaluate(x_test, y_test, transform=True)

        if error < best_error:
            best_error = error
            best_node = i
            predictions = predictions
        print("number of hidden units: {0} and test Error: {1}".format(i, error))

    print("best error {0}, best node {1}".format(best_error, best_node))
    Utils.plot_pred_actual(predictions, y_test, '')




if __name__ == "__main__":

    experiment_rbf_with_noise()