import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


np.random.seed(421)
# mean parameters
class_means = np.array([-2.0, +1.0, +3.0])
# standard deviation parameters
class_deviations = np.array([2.0, 1.2, 1.6])
# sample sizes
class_sizes = np.array([20, 15, 25])


# generate random samples
points1 = np.random.normal(loc = class_means[0],
                           scale = class_deviations[0],
                           size = class_sizes[0])
points2 = np.random.normal(loc = class_means[1],
                           scale = class_deviations[1],
                           size = class_sizes[1])
points3 = np.random.normal(loc = class_means[2],
                           scale = class_deviations[2],
                           size = class_sizes[2])
points = np.concatenate((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]),
                    np.repeat(2, class_sizes[1]),
                    np.repeat(3, class_sizes[2])))


# write data to a file
np.savetxt("data_set.csv", np.stack((points, y), axis = 1), fmt = "%f,%d")


data_interval = np.linspace(-8.0, +8.0, 1601)
density1 = stats.norm.pdf(data_interval,
                          loc = class_means[0],
                          scale = class_deviations[0])
density2 = stats.norm.pdf(data_interval,
                          loc = class_means[1],
                          scale = class_deviations[1])
density3 = stats.norm.pdf(data_interval,
                          loc = class_means[2],
                          scale = class_deviations[2])

plt.figure(figsize = (6, 4))
# plot data points of the first class
plt.plot(points1, np.repeat(-0.01, class_sizes[0]), "r|", markersize = 5)
# plot density of the first class
plt.plot(data_interval, density1, "r")
# plot data points of the second class
plt.plot(points2, np.repeat(-0.02,class_sizes[1]), "g|", markersize = 5)
# plot density of the second class
plt.plot(data_interval, density2, "g")
# plot data points of the third class
plt.plot(points3, np.repeat(-0.03, class_sizes[2]), "b|", markersize = 5)
# plot density of the third class
plt.plot(data_interval, density3, "b")
plt.show()


# read data into memory
data_set = np.genfromtxt("data_set.csv", delimiter = ",")

# get x and y values
x_train = data_set[:, 0]
y_train = data_set[:, 1].astype(int)

# get number of classes and number of samples
K = np.max(y_train)
N = data_set.shape[0]


point_colors = np.array(["red", "green", "blue"])
minimum_value = -8.0
maximum_value = +8.0
data_interval = np.linspace(minimum_value, maximum_value, 1601)


# histogram estimator

bin_width = 0.5
left_borders = np.arange(start = minimum_value,
                         stop = maximum_value,
                         step = bin_width)
right_borders = np.arange(start = minimum_value + bin_width,
                          stop = maximum_value + bin_width,
                          step = bin_width)
p_hat = np.asarray([np.sum((left_borders[b] < x_train) & (x_train <= right_borders[b]))
                    for b in range(len(left_borders))]) / (N * bin_width)

plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.01, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.02, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.03, np.sum(y_train == 3)), "b|", markersize = 5)
for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat[b], p_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat[b], p_hat[b + 1]], "k-")    
plt.show()





# naive estimator

bin_width = 0.5
p_hat = np.asarray([np.sum(((x - 0.5 * bin_width) < x_train) &
                            (x_train <= (x + 0.5 * bin_width)))
                    for x in data_interval]) / (N * bin_width)

plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.01, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.02, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.03, np.sum(y_train == 3)), "b|", markersize = 5)
plt.plot(data_interval, p_hat, "k-")
plt.show()





# kernel estimator

bin_width = 0.5
p_hat = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * \
                           np.exp(-0.5 * (x - x_train)**2 / bin_width**2))
                    for x in data_interval]) / (N * bin_width)

plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.01, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.02, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.03, np.sum(y_train == 3)), "b|", markersize = 5)
plt.plot(data_interval, p_hat, "k-")
plt.show()





# k-NN estimator

k = 11
p_hat = np.asarray([k / (2 * N * np.sort(np.abs(x - x_train))[k - 1])
                    for x in data_interval])

plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.01, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.02, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.03, np.sum(y_train == 3)), "b|", markersize = 5)
plt.plot(data_interval, p_hat, "k-")
plt.show()





# non-parametric classification

bin_width = 0.5
scores = np.zeros((len(data_interval), K))
for c in range(K):
    scores[:, c] = np.asarray([np.sum(1.0 / np.sqrt(2 * math.pi) * \
                                     np.exp(-0.5 * (x - x_train[y_train == (c + 1)])**2 / bin_width**2)) 
                               for x in data_interval]) / (N * bin_width)
    
plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.01, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.02, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.03, np.sum(y_train == 3)), "b|", markersize = 5)
for c in range(K):
    plt.plot(data_interval, scores[:, c], color = point_colors[c])
plt.scatter(data_interval, 
            np.repeat(np.max(scores) * 1.1, len(data_interval)), 
            c = point_colors[np.argmax(scores, axis = 1)],
            marker = ".")    
plt.show()





# k-NN classifier

k = 11
scores = np.zeros((len(data_interval), K))
for c in range(K):
    scores[:, c] = np.asarray([np.sum(y_train[np.argsort(np.abs(x - x_train))[range(k)]] == (c + 1)) 
                               for x in data_interval]) / k
    
plt.figure(figsize = (6, 4))
plt.plot(x_train[y_train == 1], np.repeat(-0.1, np.sum(y_train == 1)), "r|", markersize = 5)
plt.plot(x_train[y_train == 2], np.repeat(-0.2, np.sum(y_train == 2)), "g|", markersize = 5)
plt.plot(x_train[y_train == 3], np.repeat(-0.3, np.sum(y_train == 3)), "b|", markersize = 5)
for c in range(K):
    plt.plot(data_interval, scores[:, c], color = point_colors[c])
plt.scatter(data_interval,
            np.repeat(np.max(scores) * 1.1, len(data_interval)),
            c = point_colors[np.argmax(scores, axis = 1)],
            marker = ".")    
plt.show()