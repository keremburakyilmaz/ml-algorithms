import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1]
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.6
maximum_value = 5.1
x_interval = np.arange(start = minimum_value, stop = maximum_value, step = 0.001)

def plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(x_train, y_train, "b.", markersize = 10)
    plt.plot(x_test, y_test, "r.", markersize = 10)
    plt.plot(x_interval, y_interval_hat, "k-")
    plt.xlim([1.55, 5.15])
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)


def regressogram(x_query, x_train, y_train, left_borders, right_borders):
    y_hat = np.zeros(len(x_query))

    for i in range(len(x_query)):
        x = x_query[i]
        bin_index = np.where((left_borders <= x) & (x < right_borders))[0]
        if (len(bin_index)) > 0:
            bin_mask = (x_train > left_borders[bin_index[0]]) & (x_train <= right_borders[bin_index[0]])
            if np.sum(bin_mask) > 0:
                y_hat[i] = np.mean(y_train[bin_mask])
            else:
                y_hat[i] = 0

    return(y_hat)
    
bin_width = 0.35
left_borders = np.arange(start = minimum_value, stop = maximum_value, step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width, stop = maximum_value + bin_width, step = bin_width)

y_interval_hat = regressogram(x_interval, x_train, y_train, left_borders, right_borders)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)

y_test_hat = regressogram(x_test, x_train, y_train, left_borders, right_borders)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Regressogram => RMSE is {} when h is {}".format(rmse, bin_width))


def running_mean_smoother(x_query, x_train, y_train, bin_width):
    y_hat = np.zeros(len(x_query))

    for i in range(len(x_query)):
        x = x_query[i]
        w = np.abs(x_train - x) < (bin_width / 2)
        if np.sum(w) > 0:
            y_hat[i] = np.sum(y_train[w]) / np.sum(w)
        else:
            y_hat[i] = 0
    return(y_hat)

bin_width = 0.35

y_interval_hat = running_mean_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)

y_test_hat = running_mean_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Running Mean Smoother => RMSE is {} when h is {}".format(rmse, bin_width))


def kernel_smoother(x_query, x_train, y_train, bin_width):
    y_hat = np.zeros(len(x_query))
    for i in range(len(x_query)):
        x = x_query[i]
        w = np.exp(-0.5 * ((x_train - x) / bin_width) ** 2) / np.sqrt(2 * math.pi)
        y_hat[i] = np.sum(w * y_train) / np.sum(w)
        
    return(y_hat)

bin_width = 0.35

y_interval_hat = kernel_smoother(x_interval, x_train, y_train, bin_width)
fig = plot_figure(x_train, y_train, x_test, y_test, x_interval, y_interval_hat)

y_test_hat = kernel_smoother(x_test, x_train, y_train, bin_width)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("Kernel Smoother => RMSE is {} when h is {}".format(rmse, bin_width))
