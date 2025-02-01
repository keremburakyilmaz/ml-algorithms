import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.spatial.distance as dt


# define Gaussian kernel function
def gaussian_kernel(X1, X2, s):
    Distance = dt.cdist(X1, X2)
    K = np.exp(-Distance**2 / (2 * s**2))
    return(K)


np.random.seed(421)
# mean parameters
class_means = np.array([[-4.0, +4.0],
                        [+4.0, -4.0],
                        [+2.0, +2.0],
                        [-2.0, -2.0],
                        [-4.0, -4.0],
                        [+4.0, +4.0],
                        [-2.0, +2.0],
                        [+2.0, -2.0]])
# covariance parameters
class_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +0.4]],
                              [[+0.8, -0.6], 
                               [-0.6, +0.8]],
                              [[+0.8, -0.6], 
                               [-0.6, +0.8]],                              
                              [[+0.4, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +0.4]],
                              [[+0.8, +0.6], 
                               [+0.6, +0.8]],
                              [[+0.8, +0.6], 
                               [+0.6, +0.8]]])
# sample sizes
class_sizes = np.array([100, 100])


# generate random samples
points1 = np.random.multivariate_normal(mean = class_means[0, :],
                                        cov = class_covariances[0, :, :],
                                        size = class_sizes[0] // 4)
points2 = np.random.multivariate_normal(mean = class_means[1, :],
                                        cov = class_covariances[1, :, :],
                                        size = class_sizes[0] // 4)
points3 = np.random.multivariate_normal(mean = class_means[2, :],
                                        cov = class_covariances[2, :, :],
                                        size = class_sizes[0] // 4)
points4 = np.random.multivariate_normal(mean = class_means[3, :],
                                        cov = class_covariances[3, :, :],
                                        size = class_sizes[0] // 4)
points5 = np.random.multivariate_normal(mean = class_means[4, :],
                                        cov = class_covariances[4, :, :],
                                        size = class_sizes[1] // 4)
points6 = np.random.multivariate_normal(mean = class_means[5, :],
                                        cov = class_covariances[5, :, :],
                                        size = class_sizes[1] // 4)
points7 = np.random.multivariate_normal(mean = class_means[6, :],
                                        cov = class_covariances[6, :, :],
                                        size = class_sizes[1] // 4)
points8 = np.random.multivariate_normal(mean = class_means[7, :],
                                        cov = class_covariances[7, :, :],
                                        size = class_sizes[1] // 4)
X = np.vstack((points1, points2, points3, points4, points5, points6, points7, points8))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(0, class_sizes[1])))


# write data to a file
np.savetxt("data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")


# plot data points generated
plt.figure(figsize = (8, 8))
plt.plot(X[y == 1, 0], X[y == 1, 1], "r.", markersize = 10)
plt.plot(X[y == 0, 0], X[y == 0, 1], "b.", markersize = 10)
plt.xlim((-6, +6))
plt.ylim((-6, +6))
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()


# read data into memory
data_set = np.genfromtxt("data_set.csv", delimiter = ",")

# get number of samples
N_train = data_set.shape[0]
# get number of features
D = data_set.shape[1] - 1

# get X and y values
X_train = data_set[:, 0:D]
y_train = 2 * data_set[:, D:(D + 1)].astype(int) - 1


# calculate Gaussian kernel
s = 1
K_train = gaussian_kernel(X_train, X_train, s)
yyK = np.matmul(y_train, y_train.T) * K_train

# set learning parameters
C = 10
epsilon = 0.001

P = cvx.matrix(yyK)
q = cvx.matrix(-np.ones((N_train, 1)))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(1.0 * y_train.T)
b = cvx.matrix(0.0)

# use cvxopt library to solve QP problems
result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], N_train)
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C

# find bias parameter
support_indices, = np.where(alpha != 0)
active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
w0 = np.mean(y_train[active_indices] * (1 - np.matmul(yyK[active_indices][:, support_indices], alpha[support_indices])))


# calculate predictions on training samples
f_predicted = np.matmul(K_train, y_train * alpha[:,None]) + w0

# calculate confusion matrix
y_predicted = 2 * (f_predicted > 0.0) - 1
confusion_matrix = pd.crosstab(np.reshape(y_predicted, N_train), y_train.T,
                               rownames = ["y_predicted"], colnames = ["y_train"])
print(confusion_matrix)


# evaluate discriminant function on a grid
x1_interval = np.linspace(-6, +6, 101)
x2_interval = np.linspace(-6, +6, 101)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_test = np.transpose(np.vstack((x1_grid.flatten(), x2_grid.flatten())))
K_test = gaussian_kernel(X_test, X_train, s)
discriminant_values = np.reshape(np.matmul(K_test, y_train * alpha[:,None]) + w0, x1_grid.shape)

plt.figure(figsize = (8, 8))
plt.plot(X_train[y_train[:, 0] == +1, 0], X_train[y_train[:, 0] == +1, 1], "r.", markersize = 10)
plt.plot(X_train[y_train[:, 0] == -1, 0], X_train[y_train[:, 0] == -1, 1], "b.", markersize = 10)
plt.plot(X_train[support_indices, 0], X_train[support_indices, 1], "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values, levels = [-1, 0, +1], colors = "k", linestyles = ["dashed", "solid", "dashed"])
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()