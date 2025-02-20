import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dt


def linear_kernel(X1, X2):
    K = X1 @ X2.T
    return(K)

def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)


np.random.seed(421)
class_means = np.array([[+4.0, +4.0],
                        [-1.5, +0.5],
                        [+1.5, -1.5]])
class_covariances = np.array([[[+0.1, 0.0],
                               [0.0, +0.1]],
                              [[+0.2, 0.0],
                               [0.0, +0.2]],
                              [[+0.2, 0.0],
                               [0.0, +0.2]]])
class_sizes = np.array([2, 149, 149])


points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2, points3))

np.savetxt("dataset.csv", X, fmt = "%f,%f")

plt.figure(figsize = (10, 10))
plt.plot(X[:,0], X[:,1], "k.", markersize = 10)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()


data_set = np.genfromtxt("dataset.csv", delimiter = ",")

X_train = data_set[:,[0, 1]]

N_train = X_train.shape[0]
D_train = X_train.shape[1]


s = 2
K_train = gaussian_kernel(X_train, X_train, s)

C = 0.15
epsilon = 1e-3

P = cvx.matrix(K_train)
q = cvx.matrix(-0.5 * np.diag(K_train))
G = cvx.matrix(np.vstack((-np.eye(N_train), np.eye(N_train))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))
A = cvx.matrix(np.ones((1, N_train)))
b = cvx.matrix(1.0)

result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], N_train)
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C
    
support_indices, = np.where(alpha != 0)
active_indices, = np.where(np.logical_and(alpha != 0, alpha < C))
R = np.sqrt(np.matmul(alpha[support_indices], np.matmul(K_train[np.ix_(support_indices, support_indices)], alpha[support_indices])) + 
            np.mean(np.diag(K_train[np.ix_(active_indices, active_indices)])) - 2 * np.mean(np.matmul(K_train[np.ix_(active_indices, support_indices)], alpha[support_indices])))


x1_interval = np.linspace(-6, +6, 41)
x2_interval = np.linspace(-6, +6, 41)
x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
X_test = np.transpose(np.vstack((x1_grid.flatten(), x2_grid.flatten())))
K_test_train = gaussian_kernel(X_test, X_train, s)
K_test_test = gaussian_kernel(X_test, X_test, s)
discriminant_values = np.reshape(R**2 - np.matmul(alpha[support_indices], np.matmul(K_train[np.ix_(support_indices, support_indices)], 
                                                                                    alpha[support_indices])) + 2 * np.matmul(K_test_train, alpha) - np.diag(K_test_test), x1_grid.shape)

plt.figure(figsize = (10, 10))
plt.plot(X_train[:,0], X_train[:,1], "r.", markersize = 10)
plt.plot(X_train[support_indices, 0], X_train[support_indices, 1],
         "ko", markersize = 12, fillstyle = "none")
plt.contour(x1_grid, x2_grid, discriminant_values,
            levels = 0, colors = "k", linestyles = ["dashed", "solid", "dashed"])
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()