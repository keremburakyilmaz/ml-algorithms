import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.random.seed(421)
# mean parameters
class_means = np.array([-3.0, -1.0, 3.0])
# standard deviation parameters
class_deviations = np.array([1.2, 1.0, 1.3])
# sample sizes
class_sizes = np.array([40, 30, 50])


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
np.savetxt(fname = "data_set.csv",
           X = np.stack((points, y), axis = 1),
           fmt = "%f,%d")


data_interval = np.linspace(start = -7, stop = +7, num = 701)
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
plt.plot(points2, np.repeat(-0.02, class_sizes[1]), "g|", markersize = 5)
# plot density of the second class
plt.plot(data_interval, density2, "g")
# plot data points of the third class
plt.plot(points3, np.repeat(-0.03, class_sizes[2]), "b|", markersize = 5)
# plot density of the third class
plt.plot(data_interval, density3, "b")
plt.xlabel("$x$")
plt.ylabel("density")
plt.show()


# read data into memory
data_set = np.genfromtxt(fname = "data_set.csv", delimiter = ",")

# get x and y values
x = data_set[:, 0]
y = data_set[:, 1].astype(int)

# get number of classes and number of samples
K = np.max(y)
N = data_set.shape[0]


# calculate sample means
sample_means = [np.mean(x[y == (c + 1)]) for c in range(K)]
# calculate sample deviations
sample_deviations = [np.sqrt(np.mean((x[y == (c + 1)] - sample_means[c])**2))
                     for c in range(K)]
# calculate prior probabilities
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
# evaluate score functions
data_interval = np.linspace(start = -7, stop = +7, num = 701)
score_values = np.stack([- 0.5 * np.log(2 * math.pi * sample_deviations[c]**2) 
                         - 0.5 * (data_interval - sample_means[c])**2 / sample_deviations[c]**2 
                         + np.log(class_priors[c])
                         for c in range(K)])

# calculate log posteriors
log_posteriors = score_values - [np.max(score_values[:, r]) + 
                                 np.log(np.sum(np.exp(score_values[:, r] - np.max(score_values[:, r]))))
                                 for r in range(score_values.shape[1])]

plt.figure(figsize = (6, 4))
# plot score function of the first class
plt.plot(data_interval, score_values[0, :], "r")
# plot score function of the second class
plt.plot(data_interval, score_values[1, :], "g")
# plot score function of the third class
plt.plot(data_interval, score_values[2, :], "b")
plt.xlabel("$x$")
plt.ylabel("score")
plt.show()


plt.figure(figsize = (6, 4))
# plot posterior probability of the first class
plt.plot(data_interval, np.exp(log_posteriors[0, :]), "r")
# plot posterior probability of the second class
plt.plot(data_interval, np.exp(log_posteriors[1, :]), "g")
# plot posterior probability of the third class
plt.plot(data_interval, np.exp(log_posteriors[2, :]), "b")

class_assignments = np.argmax(score_values, axis = 0)

#plot region where the first class has the highest probability
plt.plot(data_interval[class_assignments == 0], 
         np.repeat(-0.05, np.sum(class_assignments == 0)), "r|")
#plot region where the second class has the highest probability
plt.plot(data_interval[class_assignments == 1], 
         np.repeat(-0.10, np.sum(class_assignments == 1)), "g|")
#plot region where the third class has the highest probability
plt.plot(data_interval[class_assignments == 2], 
         np.repeat(-0.15, np.sum(class_assignments == 2)), "b|")

plt.xlabel("$x$")
plt.ylabel("probability")

plt.show()
