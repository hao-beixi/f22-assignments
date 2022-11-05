import sys
import train_utils
import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import load_model
from mpl_toolkits.mplot3d import Axes3D

def split_data(input, target, train_percent):
    """
    Split the input and target data into two sets
    :param input: inputs [Nx2] matrix
    :param target: target [Nx1] matrix
    :param train_percent: percentage of the data that should be assigned to training
    :return: train_input, train_target, test_input, test_target
    """
    assert input.shape[0] == target.shape[0], \
        "Number of inputs and targets do not match ({} vs {})".format(input.shape[0], target.shape[0])

    indices = list(range(input.shape[0]))
    np.random.shuffle(indices)

    num_train = int(input.shape[0]*train_percent)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    return input[train_indices, :], target[train_indices,:], input[test_indices,:], target[test_indices,:]

def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev


def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev


def compute_normalization_parameters(data):
    """
    Compute normalization parameters (mean, st. dev.)
    :param data: matrix with data organized by rows [N x num_variables]
    :return: mean and standard deviation per variable as row matrices of dimension [1 x num_variables]
    """

    mean = np.mean(data, axis=0)
    stdev = np.std(data, axis=0)

    # transpose mean and stdev in case they are (2,) arrays
    if len(mean.shape) == 1:
        mean = np.reshape(mean, (1,mean.shape[0]))
    if len(stdev.shape) == 1:
        stdev = np.reshape(stdev, (1,stdev.shape[0]))

    return mean, stdev

def build_nonlinear_model(num_inputs):
    """
    Build NN model with Keras
    :param num_inputs: number of input features for the model
    :return: Keras model
    """
    input = tf.keras.layers.Input(shape=(num_inputs,), name="inputs")

    hidden1 = tf.keras.layers.Dense(256, use_bias=True, activation=tf.nn.relu)(input)

    hidden2 = tf.keras.layers.Dense(128, use_bias=True, activation=tf.nn.relu)(hidden1)

    hidden3 = tf.keras.layers.Dense(128, use_bias=True, activation=tf.nn.relu)(hidden2)

    hidden4 = tf.keras.layers.Dense(128, use_bias=True, activation=tf.nn.relu)(hidden3)

    hidden5 = tf.keras.layers.Dense(128, use_bias=True, activation=tf.nn.relu)(hidden4)

    hidden6 = tf.keras.layers.Dense(64, use_bias=True, activation=tf.nn.relu)(hidden5)

    hidden7 = tf.keras.layers.Dense(64, use_bias=True, activation=tf.nn.relu)(hidden6)


    output = tf.keras.layers.Dense(1, use_bias=True)(hidden7)

    model = tf.keras.models.Model(inputs=input, outputs=output, name="monkey_model")
    model.save('my_model.h5')
    model.save_weights('my_weights.h5')
    return model


def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # normalize
    norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='mse',
                 metrics=['mae'])

    # TODO - Create callbacks for saving checkpoints and visualizing loss on TensorBoard
    # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_monkey_weights.h5'),
                                                            monitor='mae',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)

    # do training for thfe specified number of epochs and with the given batch size
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
              validation_data=(norm_val_input, val_target),
              callbacks=[tbCallBack, checkpointCallBack]) # add this extra parameter to the fit function
    # do training for the specified number of epochs and with the given batch size
    # TODO - Add callbacks to fit funciton
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(norm_val_input, val_target))


def test_model(model, test_input, input_mean, input_stdev, batch_size=60):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # normalize
    norm_test_input = normalize_data_per_row(test_input, input_mean, input_stdev)

    # evaluate
    predicted_targets = model.predict(norm_test_input, batch_size=batch_size)

    return predicted_targets


def compute_average_L2_error(test_target, predicted_targets):
    """
    Compute the average L2 error for the predictions
    :param test_target: matrix with ground truth targets [N x 1]
    :param predicted_targets: matrix with predicted targets [N x 1]
    :return: average L2 error
    """
    diff = predicted_targets - test_target
    l2_err = np.sqrt(np.sum(np.power(diff, 2), axis=1))
    assert l2_err.shape[0] == predicted_targets.shape[0], \
        "Invalid dim {} vs {}".format(l2_err.shape, predicted_targets.shape)
    average_l2_err = np.mean(l2_err)

    return average_l2_err


def normalize_data_per_row(data, mean, stdev):
    """
    Normalize a give matrix of data (samples must be organized per row)
    :param data: input data
    :param mean: mean for normalization
    :param stdev: standard deviation for normalization
    :return: whitened data, (data - mean) / stdev
    """

    # sanity checks!
    assert len(data.shape) == 2, "Expected the input data to be a 2D matrix"
    assert data.shape[1] == mean.shape[1], "Data - Mean size mismatch ({} vs {})".format(data.shape[1], mean.shape[1])
    assert data.shape[1] == stdev.shape[1], "Data - StDev size mismatch ({} vs {})".format(data.shape[1], stdev.shape[1])

    centered = data - np.tile(mean, (data.shape[0], 1))
    normalized_data = np.divide(centered, np.tile(stdev, (data.shape[0],1)))

    return normalized_data


def train_model(model, train_input, train_target, val_input, val_target, input_mean, input_stdev,
                epochs=20, learning_rate=0.01, batch_size=16):
    """
    Train the model on the given data
    :param model: Keras model
    :param train_input: train inputs
    :param train_target: train targets
    :param val_input: validation inputs
    :param val_target: validation targets
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :param epochs: epochs for gradient descent
    :param learning_rate: learning rate for gradient descent
    :param batch_size: batch size for training with gradient descent
    """

    # normalize
    norm_train_input = normalize_data_per_row(train_input, input_mean, input_stdev)
    norm_val_input = normalize_data_per_row(val_input, input_mean, input_stdev)

    # compile the model: define optimizer, loss, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                 loss='mse',
                 metrics=['mae'])

    # TODO - Create callbacks for saving checkpoints and visualizing loss on TensorBoard
    # tensorboard callback
    logs_dir = 'logs/log_{}'.format(datetime.datetime.now().strftime("%m-%d-%Y-%H-%M"))
    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=logs_dir, write_graph=True)

    # save checkpoint callback
    checkpointCallBack = tf.keras.callbacks.ModelCheckpoint(os.path.join(logs_dir,'best_monkey_weights.h5'),
                                                            monitor='mae',
                                                            verbose=0,
                                                            save_best_only=True,
                                                            save_weights_only=False,
                                                            mode='auto',
                                                            save_freq=1)

    # do training for thfe specified number of epochs and with the given batch size
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
              validation_data=(norm_val_input, val_target),
              callbacks=[tbCallBack, checkpointCallBack]) # add this extra parameter to the fit function
    # do training for the specified number of epochs and with the given batch size
    # TODO - Add callbacks to fit funciton
    model.fit(norm_train_input, train_target, epochs=epochs, batch_size=batch_size,
             validation_data=(norm_val_input, val_target))


def test_model(model, test_input, input_mean, input_stdev, batch_size=16):
    """
    Test a model on a given data
    :param model: trained model to perform testing on
    :param test_input: test inputs
    :param input_mean: mean for the variables in the inputs (for normalization)
    :param input_stdev: st. dev. for the variables in the inputs (for normalization)
    :return: predicted targets for the given inputs
    """
    # normalize
    norm_test_input = normalize_data_per_row(test_input, input_mean, input_stdev)

    # evaluate
    predicted_targets = model.predict(norm_test_input, batch_size=batch_size)

    return predicted_targets


def compute_average_L2_error(test_target, predicted_targets):
    """
    Compute the average L2 error for the predictions
    :param test_target: matrix with ground truth targets [N x 1]
    :param predicted_targets: matrix with predicted targets [N x 1]
    :return: average L2 error
    """
    diff = predicted_targets - test_target
    l2_err = np.sqrt(np.sum(np.power(diff, 2), axis=1))
    assert l2_err.shape[0] == predicted_targets.shape[0], \
        "Invalid dim {} vs {}".format(l2_err.shape, predicted_targets.shape)
    average_l2_err = np.mean(l2_err)

    return average_l2_err

def plot_2D_function_data(ax, inputs, targets, label, color='k'):
    """
    Method that generates scatter plot of inputs and targets
    :param inputs: inputs [Nx2] matrix
    :param targets: target [Nx1] matrix
    :param label: label for the legend
    :param color: points color
    """
    ax.scatter(inputs[:,0], inputs[:, 1], targets, s=10, c=color, label=label)


def plot_train_and_test(train_input, train_target, test_input, test_target, train_label, test_label, title="Scatter Plot"):
    """
    Method that plots two sets of data (train and test)
    :param train_input: inputs [Nx2] matrix
    :param train_target: target [Nx1] matrix
    :param test_input: inputs [Nx2] matrix
    :param test_target: target [Nx1] matrix
    :param train_label: label for the train data
    :param test_label: label for the test data
    :param title: plot title
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    plot_2D_function_data(ax, train_input, train_target, train_label, 'k')
    plot_2D_function_data(ax, test_input, test_target, test_label, 'r')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    ax.legend()
    plt.show()


def plot_test_predictions(test_input, test_target, predicted_targets, title="Predictions"):
    """
    Method that plots the ground truth targets and predictions
    :param test_input: input values as an [Nx2] matrix
    :param test_target: ground truth target values as a [Nx1] matrix
    :param predicted_targets: predicted targets as a [Nx1] matrix
    :param title: plot title
    """
    fig = plt.figure(figsize=(16, 4))

    ax = fig.add_subplot(131, projection='3d')
    plot_2D_function_data(ax, test_input, test_target, "Ground Truth", 'b')
    plot_2D_function_data(ax, test_input, predicted_targets, "Predicted", 'r')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(title)

    ax = fig.add_subplot(132)
    ax.scatter(test_input[:,0], test_target, s=10, c='b')
    ax.scatter(test_input[:,0], predicted_targets, s=10, c='r')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title('y=0 projection')

    ax = fig.add_subplot(133)
    ax.scatter(test_input[:,1], test_target, s=10, c='b')
    ax.scatter(test_input[:,1], predicted_targets, s=10, c='r')
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title('x=0 projection')

    plt.tight_layout()
    plt.show()


def main(epochs, lr, batch_size, input, target):
    """
    Main function
    :param epochs: number of epochs to train for
    :param lr: learning rate
    """

    # split data into training (70%) and testing (30%)
    all_train_input, all_train_target, test_input, test_target = split_data(input, target, 0.6)

    # visualize all training/testing (uncomment if you want to visualize the whole dataset)
    # plot_train_and_test(all_train_input, all_train_target, test_input, test_target, "train", "test", title="Train/Test Data")

    # split training data into actual training and validation
    train_input, train_target, val_input, val_target = split_data(all_train_input, all_train_target, 0.8)

    # visualize training/validation (uncomment if you want to visualize the training/validation data)

    # normalize input data and save normalization parameters to file
    mean, stdev = compute_normalization_parameters(train_input)

    # build the model
    model = build_nonlinear_model(train_input.shape[1])

    # train the model
    print("\n\nTRAINING...")
    train_model(model, train_input, train_target, val_input, val_target, mean, stdev,
                epochs=epochs, learning_rate=lr, batch_size=batch_size)

    # test the model
    print("\n\nTESTING...")
    predicted_targets = test_model(model, test_input, mean, stdev)

    # Report average L2 error
    l2_err = compute_average_L2_error(test_target, predicted_targets)
    print("L2 Error on Testing Set: {}".format(l2_err))

    # visualize the result
    #plot_test_predictions(test_input, test_target, predicted_targets, title="Predictions")


if __name__ == "__main__":
    features, targets = train_utils.load_data(str(sys.argv[1]))
    batch_size = 10
    epochs = 200
    lr = 0.00001

    main(epochs, lr, batch_size, features, targets)
