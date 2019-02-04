import Utils
import numpy as np
from RBF_NN import rbf_model


def experiment_rbf_with_noise():
    input_type = 'sin'
    np.random.seed(123)
    noise = 0.1
    lr = 1
    n_epochs = 300
    batch_train = False
    cov = 0.2

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    predictions_ = []
    best_error = 100
    best_node = -1

    for i in np.arange(1, 30, 1):

        mean = Utils.compute_rbf_centers(i)
        # mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, eta=0.2, iterations=100,threshold=1)

        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)

        model.fit()

        predictions = model.forward_pass(x_test, True)
        error = model.evaluate(x_test, y_test, transform=True)

        if error < best_error:
            best_error = error
            best_node = i
            predictions_ = predictions
        print("best error {0}, best node {1}".format(best_error, best_node))
    print("best error {0}, best node {1}".format(best_error, best_node))
    Utils.plot_pred_actual(predictions_, y_test, 'sequential noisy RBF , sin(2x) lr : {0}, sigma : {1}'.format(lr,cov))


# without noise best nodes = 61
# with noise best nodes = 10
def experiment_competitive_learning():
    input_type = 'sin'
    np.random.seed(123)
    num_hidden_units = 5
    noise = 0.1
    lr_ = 0.2
    n_epochs = 200
    batch_train = False

    cov = 1
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)


    [predictions, error] = run_rbf_expe(units=20, noise=noise, lr=lr_,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=False)

    [predictions_competitive, error_competitive] = run_rbf_expe(units=10, noise=noise, lr=lr_,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=True)


    print("number of hidden units: {0} test Error: {1}".format(num_hidden_units, error))
    print("competitive number of hidden units: {0} test Error: {1}".format(num_hidden_units, error_competitive))


    error = [predictions, predictions_competitive]
    legend = ['Manual init', 'Competitive init', 'Actual']
    Utils.plot_many_lines(error, y_test, legend, 'RBF network with noise 0.1 on data,  nodes : {0} , lr : {1} , epochs : {2}'
                                .format(num_hidden_units,lr_, n_epochs))



def run_rbf_expe( units = 9, noise = 0, lr = 0 , epochs = 500 , batch = True, cov = 0.3, competitive = True):
    input_type = 'sin'
    np.random.seed(123)
    num_hidden_units = units
    n_epochs = epochs
    batch_train = batch
    cov = cov


    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    if competitive:
        mean_competitive = Utils.compute_rbf_centers_competitive_learning(x_train, num_hidden_units, eta=0.2, iterations=600)
    else:
        mean_competitive = Utils.compute_rbf_centers(num_hidden_units)

    model_competitive = rbf_model(x_train, y_train, mean_competitive, cov, n_epochs, num_hidden_units, lr, batch_train=batch_train)
    model_competitive.fit()

    predictions_competitive = model_competitive.forward_pass(x_test, True)
    error_competitive = model_competitive.evaluate(x_test, y_test, transform=True)
    return [predictions_competitive, error_competitive]

def experiment_batch_without_noise():
    input_type = 'sin'
    np.random.seed(123)
    lr = 0.2
    n_epochs = 300
    batch_train = True

    cov = 1

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type)

    predictions = []
    best_error = 100
    best_node = -1
    for i in np.arange(1, len(x_train), 1):

        mean = Utils.compute_rbf_centers(i)

        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)

        model.fit()

        predictions = model.forward_pass(x_test, True)
        error = model.evaluate(x_test, y_test, transform=True)

        if error < best_error:
            best_error = error
            best_node = i
            predictions = predictions
        print("number of hidden units: {0} test Error: {1}".format(i, error))

    print("best error {0}, best node {1}".format(best_error, best_node))
    Utils.plot_pred_actual(predictions, y_test, '')



def experiment_competitive_learning_vanilla_comparison():
    input_type = 'sin'
    np.random.seed(123)
    num_hidden_units = 5
    noise = 0.1
    lr = 1
    n_epochs = 500
    batch_train = False

    cov = 1
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)


    [predictions, error] = run_rbf_expe(units=5, noise=0.1, lr=lr,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=False)

    [predictions_competitive, error_competitive] = run_rbf_expe(units=5, noise=0.1, lr=lr,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=True)


    print("number of hidden units: {0} test Error: {1}".format(num_hidden_units, error))
    print("competitive number of hidden units: {0} test Error: {1}".format(num_hidden_units, error_competitive))


    error = [predictions, predictions_competitive]
    legend = ['Manual init', 'Competitive init', 'Actual']
    Utils.plot_many_lines(error, y_test, legend, 'RBF network with noise 0.1 on data,  nodes : {0} , lr : {1} , epochs : {2}'
                                .format(num_hidden_units,lr, n_epochs))




def experiment_plot_error_with_nodes():
    input_type = 'sin'
    np.random.seed(123)
    noise = 0.1
    lr = 0.01
    n_epochs = 300
    batch_train = False
    cov = 0.4

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    num_nodes = 50
    predictions_ = []
    best_error = 100
    best_node = -1
    errors = np.zeros(num_nodes)
    for i in np.arange(1, num_nodes, 1):

        mean = Utils.compute_rbf_centers(i)
        # mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, eta=0.2, iterations=100,threshold=1)

        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)

        model.fit()

        predictions = model.forward_pass(x_test, True)
        error = model.evaluate(x_test, y_test, transform=True)
        errors[i] = error
        if error < best_error:
            best_error = error
            best_node = i
            predictions_ = predictions
        print("best error {0}, best node {1}".format(best_error, best_node))
    print("best error {0}, best node {1}".format(best_error, best_node))
    # Utils.plot_pred_actual(predictions_, y_test, '')
    Utils.plot_error_nodes(errors, num_nodes, ['error'],'seq noisy RBF , sin(2x) lr : {0}, sigma : {1}'.format(lr,cov))


if __name__ == "__main__":
    # experiment_plot_error_with_nodes()

    # experiment_batch_without_noise()

    # experiment_rbf_with_noise()

    experiment_competitive_learning()

    # experiment_competitive_learning_vanilla_comparison()