import cvxopt as cvx
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as dt

def gaussian_kernel(X1, X2, s):
    D = dt.cdist(X1, X2)
    K = np.exp(-D**2 / (2 * s**2))
    return(K)

data_set = np.genfromtxt("dataset.csv", delimiter = ",")

np.random.seed(421)
train_indices = np.random.choice(range(data_set.shape[0]), size = 100, replace = False)
test_indices = np.setdiff1d(range(data_set.shape[0]), train_indices)
X_train = data_set[train_indices, 0:1]
y_train = data_set[train_indices, 1]
X_test = data_set[test_indices, 0:1]
y_test = data_set[test_indices, 1]

N_train = len(y_train)
D_train = X_train.shape[1]

s = 8
K_train = gaussian_kernel(X_train, X_train, s)

C = 1000
tube = 8
epsilon = 1e-3

P = cvx.matrix(np.vstack((np.hstack((+K_train, -K_train)), np.hstack((-K_train, +K_train)))))
q = cvx.matrix(np.append(tube - y_train, tube + y_train)[:,None])
A = cvx.matrix(np.hstack((+1.0 * np.ones((1, N_train)), -1.0 * np.ones((1, N_train)))))
b = cvx.matrix(0.0)
G = cvx.matrix(np.vstack((np.hstack((-np.eye(N_train), np.zeros((N_train, N_train)))),
                          np.hstack((+np.eye(N_train), np.zeros((N_train, N_train)))),
                          np.hstack((np.zeros((N_train, N_train)), -np.eye(N_train))),
                          np.hstack((np.zeros((N_train, N_train)), +np.eye(N_train))))))
h = cvx.matrix(np.vstack((np.zeros((N_train, 1)), C * np.ones((N_train, 1)),
                          np.zeros((N_train, 1)), C * np.ones((N_train, 1)))))

result = cvx.solvers.qp(P, q, G, h, A, b)
alpha = np.reshape(result["x"], 2 * N_train)
alpha[alpha < C * epsilon] = 0
alpha[alpha > C * (1 - epsilon)] = C
alpha = alpha[np.arange(0, N_train)] - alpha[np.arange(N_train, 2 * N_train)]

support_indices, = np.where(alpha != 0)
active_indices, = np.where(np.logical_and(alpha != 0, np.abs(alpha) < C))
w0 = np.mean(y_train[active_indices] - tube * np.sign(alpha[active_indices])) - np.mean(np.matmul(K_train[np.ix_(active_indices, support_indices)], alpha[support_indices]))



y_predicted = np.matmul(K_train, alpha[:,None]) + w0

rmse_train = np.sqrt(np.mean((y_predicted.flatten() - y_train)**2))
print(rmse_train)

K_test = gaussian_kernel(X_test, X_train, s)
y_predicted = np.matmul(K_test, alpha[:,None]) + w0

rmse_test = np.sqrt(np.mean((y_predicted.flatten() - y_test)**2))
print(rmse_test)

x_interval = np.linspace(0, 60, 601)
X_interval = x_interval[:,None]
K_interval = gaussian_kernel(X_interval, X_train, s)
fitted_values = np.matmul(K_interval, alpha[:,None]) + w0

plt.figure(figsize = (10, 6))
plt.plot(X_train, y_train, "k.", markersize = 10)
plt.plot(X_train[support_indices], y_train[support_indices],
         "ko", markersize = 12, fillstyle = "none")
plt.plot(X_test, y_test, "b.", markersize = 10)
plt.plot(X_test, y_predicted, "r.", markersize = 10)
for i in range(len(y_test)):
    plt.vlines(X_test[i], y_test[i], y_predicted[i], "r", linestyle = "dashed")
    
plt.plot(X_interval, fitted_values, "m")
plt.plot(X_interval, fitted_values - tube, "m", linestyle = "dashed")
plt.plot(X_interval, fitted_values + tube, "m", linestyle = "dashed")
plt.xlabel("x")
plt.ylabel("y")
plt.show()