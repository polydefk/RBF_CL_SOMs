import Utils
import numpy as np
from RBF_NN import rbf_model
from SOM import SOM


def experiment_rbf_with_noise():
    input_type = 'sin'
    np.random.seed(123)
    noise = 0.1
    lr = 0.1
    n_epochs = 200
    batch_train = False
    cov = 0.2

    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    predictions_ = []
    best_error = 100
    best_node = -1

    for i in np.arange(1, 49, 1):

        # mean = Utils.compute_rbf_centers(i)
        mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, eta=0.2, iterations=600,threshold=0.2)

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
    n_epochs = 100
    batch_train = False

    cov = 0.5
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)


    [predictions, error,  mean] = run_rbf_expe(units=9, noise=noise, lr=lr_,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=False)

    [predictions_competitive, error_competitive, mean_competitive] = run_rbf_expe(units=9, noise=noise, lr=lr_,
                                                                epochs=n_epochs,batch=batch_train, cov=cov, competitive=True)


    print("number of hidden units: {0} test Error: {1}".format(num_hidden_units, error))
    print("competitive number of hidden units: {0} test Error: {1}".format(num_hidden_units, error_competitive))



    Utils.plot_data_means_1D(x_train, mean, mean_competitive, 'Plot data along with means')

    error = [predictions, predictions_competitive]
    legend = ['Manual init', 'Competitive init', 'Actual']
    Utils.plot_many_lines(error, y_test, legend, 'RBF network with lr : {1} , epochs : {2}, width : {3}'
                                .format(num_hidden_units,lr_, n_epochs,cov))



def run_rbf_expe( units = 9, noise = 0, lr = 0 , epochs = 500 , batch = True, cov = 0.3, competitive = True, threshold=0):
    input_type = 'sin'
    np.random.seed(123)
    num_hidden_units = units
    n_epochs = epochs
    batch_train = batch
    cov = cov


    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    if competitive:
        mean = Utils.compute_rbf_centers_competitive_learning(x_train, num_hidden_units, eta=0.2,
                                                              iterations=600,threshold=threshold)
    else:
        mean = Utils.compute_rbf_centers(num_hidden_units)

    model_competitive = rbf_model(x_train, y_train, mean, cov, n_epochs, num_hidden_units, lr, batch_train=batch_train)
    model_competitive.fit()

    predictions_competitive = model_competitive.forward_pass(x_test, True)
    error_competitive = model_competitive.evaluate(x_test, y_test, transform=True)
    return [predictions_competitive, error_competitive, mean]

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
    lr_ = 0.2
    n_epochs = 100
    batch_train = False

    cov = 0.5
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise=noise)

    [predictions, error] = run_rbf_expe(units=5, noise=noise, lr=lr_,
                                        epochs=n_epochs, batch=batch_train, cov=cov, competitive=True, threshold = 0)

    [predictions_competitive, error_competitive] = run_rbf_expe(units=5, noise=noise, lr=lr_,
                                                                epochs=n_epochs, batch=batch_train, cov=cov,
                                                                competitive=True, threshold = 0.2)

    print("number of hidden units: {0} test Error: {1}".format(num_hidden_units, error))
    print("competitive number of hidden units: {0} test Error: {1}".format(num_hidden_units, error_competitive))

    error = [predictions, predictions_competitive]
    legend = ['Vanilla CL', 'Many winners CL', 'Actual']
    Utils.plot_many_lines(error, y_test, legend, 'CL Comparison trained with with lr : {1} , epochs : {2}, width : {3}'
                          .format(num_hidden_units, lr_, n_epochs, cov))


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


def run_comp_learning():
    input_type = 'sin'
    np.random.seed(123)
    num_hidden_units = 20
    noise = 0.1
    lr = 0.2
    n_epochs = 100
    batch_train = False

    cov = 0.5
    x_train, y_train, x_test, y_test = Utils.create_dataset(input_type, noise)

    mean = Utils.compute_rbf_centers(num_hidden_units)

    model = rbf_model(x_train, y_train, mean, cov, n_epochs, num_hidden_units, lr, batch_train=batch_train)
    model.fit()

    predictions = model.forward_pass(x_test, True)
    error = model.evaluate(x_test, y_test, transform=True)
    print(error)


def experimen_2d_data():
    x_train, y_train, x_test, y_test = Utils.load_ballist_data()

    lr = 0.1
    n_epochs = 200
    batch_train = False

    threshold = 0
    cov = 0.5

    predictions_ = []
    best_error = 100
    best_node = -1

    for i in np.arange(1, 15, 1):

        mean = Utils.compute_rbf_centers_competitive_learning(x_train, i, eta=0.2, iterations=600,
                                                              threshold=threshold)




        model = rbf_model(x_train, y_train, mean, cov, n_epochs, i, lr, batch_train=batch_train)
        model.fit()

        predictions = model.forward_pass(x_test, True)
        error = model.evaluate(x_test, y_test, transform=True)
        print(error)

        if error < best_error:
            best_error = error
            best_node = i
            predictions_ = predictions
        print("best error {0}, best node {1}".format(best_error, best_node))
    print("best error {0}, best node {1}".format(best_error, best_node))

    best_mean_cl = Utils.compute_rbf_centers_competitive_learning(x_train, best_node, eta=0.2, iterations=200,
                                                          threshold=threshold)

    best_means = Utils.compute_rbf_centers(best_node, x_train)


    # print(best_mean_cl)

    Utils.plot_data_means(x_train, best_means, best_mean_cl, 'Plot data along with means')

    Utils.plot_pred_actual(predictions_, y_test, 'RBF on 2d data lr : {0}, sigma : {1}'.format(lr, cov))





def run_animals_experiment():
    props, names = Utils.load_animals()

    # props = shuffle(props)

    weight_shape = (100, 84)
    epochs = 20
    eta = 0.2

    som = SOM(shape=weight_shape, n_epochs=epochs, eta=eta, neighbors_num=50)
    som.fit(props)

    pred = som.predict(props, names)

    print(pred)


def run_cities_experiment():
    cities_data, cities_labels = Utils.load_cities()

    weight_shape = (10, 2)
    epochs = 20
    eta = 0.2

    som = SOM(shape=weight_shape, n_epochs=epochs, eta=eta, neighbors_num=2, neighbohood_function='circular')
    som.fit(cities_data)

    pred = som.predict(cities_data, cities_labels)

    print(pred)


    Utils.plot_cities_tour(cities_data, pred)


if __name__ == "__main__":
    # run_comp_learning()


    # experiment_plot_error_with_nodes()

    # experiment_batch_without_noise()

    # experiment_rbf_with_noise()

    # experiment_competitive_learning()

    # experiment_competitive_learning_vanilla_comparison()

    # experimen_2d_data()


    ############# PART 2 ##################

    # run_animals_experiment()
    run_cities_experiment()


