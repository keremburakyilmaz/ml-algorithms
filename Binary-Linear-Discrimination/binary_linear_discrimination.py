import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def safelog(x):
    return(np.log(x + 1e-300))

# define the sigmoid function
def sigmoid(X, w, w0):
    return(1 / (1 + np.exp(-(np.matmul(X, w) + w0))))

# define the gradient functions
def gradient_w(X, y_truth, y_predicted):
    return(-np.matmul(X.T, y_truth - y_predicted))

def gradient_w0(y_truth, y_predicted):
    return(-np.sum(y_truth - y_predicted))

# set learning parameters
eta = 0.01
epsilon = 0.001

# randomly initalize w and w0
np.random.seed(421)
w = np.random.uniform(low = -0.01, high = 0.01, size = (D, 1))
w0 = np.random.uniform(low = -0.01, high = 0.01, size = 1)
# we do not give 0, 0 because it would be stuck, we need to give it a nudge


np.random.seed(421)
# mean parameters
class_means = np.array([[+1.5, +1.5],
                        [-1.5, -1.5]])
# covariance parameters
class_covariances = np.array([[[+1.6, +1.2], 
                               [+1.2, +1.6]],
                              [[+1.6, -1.2], 
                               [-1.2, +1.6]]])
# sample sizes
class_sizes = np.array([120, 180])


# generate random samples
points1 = np.random.multivariate_normal(mean = class_means[0, :],
                                        cov = class_covariances[0, :, :],
                                        size = class_sizes[0])
points2 = np.random.multivariate_normal(mean = class_means[1, :],
                                        cov = class_covariances[1, :, :],
                                        size = class_sizes[1])
X = np.vstack((points1, points2))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(0, class_sizes[1])))


# write data to a file
np.savetxt("data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")


x1_interval = np.linspace(-5, +5, 501)
x2_interval = np.linspace(-5, +5, 501)
x1_grid, x2_grid= np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

D1 = stats.multivariate_normal.pdf(X_grid, mean = class_means[0, :],
                                   cov = class_covariances[0, :, :])
D1 = D1.reshape((len(x1_interval), len(x2_interval)))
D2 = stats.multivariate_normal.pdf(X_grid, mean = class_means[1, :],
                                   cov = class_covariances[1, :, :])
D2 = D2.reshape((len(x1_interval), len(x2_interval)))

# plot data points generated
plt.figure(figsize = (6, 6))
plt.plot(points1[:, 0], points1[:, 1], "r.", markersize = 6)
plt.plot(points2[:, 0], points2[:, 1], "b.", markersize = 6)
plt.contour(x1_grid, x2_grid, D1, levels = [0.03, 0.06, 0.09, 0.12, 0.14],
            colors = "r", linestyles = "dashed")
plt.contour(x1_grid, x2_grid, D2, levels = [0.03, 0.06, 0.09, 0.12, 0.14],
            colors = "b", linestyles = "dashed")
plt.xlim((-5, +5))
plt.ylim((-5, +5))
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


# learn w and w0 using gradient descent
iteration = 1
objective_values = []
while True:
    y_predicted = sigmoid(X, w, w0) 

    objective_values = np.append(objective_values,
                                 -np.sum(y_truth * safelog(y_predicted) + 
                                         (1 - y_truth) * safelog(1 - y_predicted)))
    
    if iteration == 501:
        break   

    w_old = w
    w0_old = w0

    w = w - eta * gradient_w(X, y_truth, y_predicted)
    w0 = w0 - eta * gradient_w0(y_truth, y_predicted)

    if np.sqrt((w0 - w0_old)**2 + np.sum((w - w_old)**2)) < epsilon:
        break

    iteration = iteration + 1


# calculate confusion matrix
y_predicted = 1 * (y_predicted > 0.5)
confusion_matrix = pd.crosstab(y_predicted.T, y_truth.T, 
                               rownames = ["y_pred"], 
                               colnames = ["y_truth"])
print(confusion_matrix)


# evaluate discriminant function on a grid
x1_interval = np.linspace(-5, +5, 501)
x2_interval = np.linspace(-5, +5, 501)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T

discriminant_values = sigmoid(X_grid, w, w0)
discriminant_values = discriminant_values.reshape((len(x1_interval), len(x2_interval)))

plt.figure(figsize = (6, 6))
plt.plot(X[y_truth[:, 0] == 1, 0],  X[y_truth[:, 0] == 1, 1],
         "r.", markersize = 6)
plt.plot(X[y_truth[:, 0] == 0, 0], X[y_truth[:, 0] == 0, 1],
         "b.", markersize = 6)
plt.plot(X[y_predicted[:, 0] != y_truth[:, 0], 0], X[y_predicted[:, 0] != y_truth[:, 0], 1],
         "ko", markersize = 8, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values, levels = [0.5], colors = "k")
plt.xlim((-5, +5))
plt.ylim((-5, +5))
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.show()