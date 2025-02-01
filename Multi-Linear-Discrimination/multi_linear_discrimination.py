import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def safelog(x):
    return(np.log(x + 1e-300))

# define the softmax function
def softmax(X, W, w0):
    N = X.shape[0]
    K = W.shape[1]
    scores = np.matmul(X, W) + w0
    scores = np.exp(scores - np.max(scores, axis = 1, keepdims = True))
    scores = scores / np.sum(scores, axis = 1, keepdims = True)
    return(scores)

# define the gradient functions
def gradient_W(X, Y_truth, Y_predicted):
    return(-np.matmul(X.T, Y_truth - Y_predicted))

def gradient_w0(Y_truth, Y_predicted):
    return(-np.sum(Y_truth - Y_predicted, axis = 0))

# set learning parameters
eta = 0.01
epsilon = 0.001


np.random.seed(421)
# mean parameters
class_means = np.array([[+0.0, +1.5], 
                        [-2.5, -3.0], 
                        [+2.5, -3.0]])
# covariance parameters
class_covariances = np.array([[[+1.0, +0.2], 
                               [+0.2, +3.2]],
                              [[+1.6, -0.8], 
                               [-0.8, +1.0]],
                              [[+1.6, +0.8], 
                               [+0.8, +1.0]]])
# sample sizes
class_sizes = np.array([100, 100, 100])


# generate random samples
points1 = np.random.multivariate_normal(mean = class_means[0, :],
                                        cov = class_covariances[0, :, :], 
                                        size = class_sizes[0])
points2 = np.random.multivariate_normal(mean = class_means[1, :],
                                        cov = class_covariances[1, :, :],
                                        size = class_sizes[1])
points3 = np.random.multivariate_normal(mean = class_means[2, :],
                                        cov = class_covariances[2, :, :],
                                        size = class_sizes[2])
X = np.vstack((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]),
                    np.repeat(2, class_sizes[1]),
                    np.repeat(3, class_sizes[2])))


# write data to a file
np.savetxt("data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")


x1_interval = np.linspace(-7, +7, 701)
x2_interval = np.linspace(-7, +7, 701)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

D1 = stats.multivariate_normal.pdf(X_grid, mean = class_means[0, :],
                                   cov = class_covariances[0, :, :])
D1 = D1.reshape((len(x1_interval), len(x2_interval)))
D2 = stats.multivariate_normal.pdf(X_grid, mean = class_means[1, :],
                                   cov = class_covariances[1, :, :])
D2 = D2.reshape((len(x1_interval), len(x2_interval)))
D3 = stats.multivariate_normal.pdf(X_grid, mean = class_means[2, :],
                                   cov = class_covariances[2, :, :])
D3 = D3.reshape((len(x1_interval), len(x2_interval)))

# plot data points generated
plt.figure(figsize = (6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize = 6)
plt.plot(points2[:, 0], points2[:, 1], "g.", markersize = 6)
plt.plot(points3[:, 0], points3[:, 1], "b.", markersize = 6)
plt.contour(x1_grid, x2_grid, D1, levels = [0.01, 0.03, 0.05, 0.07],
            colors = "r", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D2, levels = [0.01, 0.03, 0.05, 0.07],
            colors = "g", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D3, levels = [0.01, 0.03, 0.05, 0.07],
            colors = "b", linestyles = "dashed",)
plt.xlim((-6.5, +6.5))
plt.ylim((-6.5, +6.5))
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# read data into memory
data_set = np.genfromtxt("data_set.csv", delimiter = ",")

# get number of samples
N = data_set.shape[0]
# get number of features
D = data_set.shape[1] - 1

# get X and y values
X = data_set[:, 0:D]
y_truth = data_set[:, D:(D + 1)].astype(int)

# get number of classes
K = np.max(y_truth)

# one-of-K encoding
Y_truth = np.zeros((N, K)).astype(int)
Y_truth[range(N), y_truth[:, 0] - 1] = 1


# randomly initalize W and w0
np.random.seed(421)
W = np.random.uniform(low = -0.01, high = 0.01, size = (D, K))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))


# learn W and w0 using gradient descent
iteration = 1
objective_values = []
while True:
    Y_predicted = softmax(X, W, w0)

    objective_values = np.append(objective_values, -np.sum(Y_truth * safelog(Y_predicted)))

    W_old = W
    w0_old = w0

    W = W - eta * gradient_W(X, Y_truth, Y_predicted)
    w0 = w0 - eta * gradient_w0(Y_truth, Y_predicted)

    if np.sqrt(np.sum((w0 - w0_old)**2) + np.sum((W - W_old)**2)) < epsilon:
        break

    iteration = iteration + 1


# calculate confusion matrix
y_predicted = np.argmax(Y_predicted, axis = 1) + 1
confusion_matrix = pd.crosstab(y_predicted, y_truth.T,
                               rownames = ["y_pred"],
                               colnames = ["y_truth"])
print(confusion_matrix)


# evaluate discriminant function on a grid
x1_interval = np.linspace(-7, +7, 701)
x2_interval = np.linspace(-7, +7, 701)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

discriminant_values = softmax(X_grid, W, w0)
discriminant_values = discriminant_values.reshape((len(x1_interval), len(x2_interval), K))

A = discriminant_values[:, :, 0]
B = discriminant_values[:, :, 1]
C = discriminant_values[:, :, 2]
A[(A < B) & (A < C)] = None
B[(B < A) & (B < C)] = None
C[(C < A) & (C < B)] = None
discriminant_values[:, :, 0] = A
discriminant_values[:, :, 1] = B
discriminant_values[:, :, 2] = C

plt.figure(figsize = (6, 6))
plt.plot(X[y_truth[:, 0] == 1, 0], X[y_truth[:, 0] == 1, 1], "r.", markersize = 6)
plt.plot(X[y_truth[:, 0] == 2, 0], X[y_truth[:, 0] == 2, 1], "g.", markersize = 6)
plt.plot(X[y_truth[:, 0] == 3, 0], X[y_truth[:, 0] == 3, 1], "b.", markersize = 6)
plt.plot(X[y_predicted != y_truth[:, 0], 0], X[y_predicted != y_truth[:, 0], 1],
         "ko", markersize = 8, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 1],
            levels = [0], colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 0] - discriminant_values[:, :, 2],
            levels = [0], colors = "k")
plt.contour(x1_grid, x2_grid, discriminant_values[:, :, 1] - discriminant_values[:, :, 2],
            levels = [0], colors = "k")
plt.xlim((-6.5, +6.5))
plt.ylim((-6.5, +6.5))
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()