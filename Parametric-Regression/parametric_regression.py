import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg

# read data into memory
data_set = np.genfromtxt(fname = "data_set.csv", delimiter = ",", skip_header = 1)

# get x and y values
x = data_set[:, 0]
y = data_set[:, 1]

# get number of samples
N = data_set.shape[0]

x_test = np.linspace(start = 0, stop = 60, num = 601)


plt.figure(figsize = (8, 6))
plt.plot(x, y, "k.", markersize = 6)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


#linear regression

# calculate A
A = np.array([[N, np.sum(x)],
              [np.sum(x), np.sum(x**2)]])
print(A)

# calculate b
b = np.array([[np.sum(y)], [np.sum(y * x)]])
print(b)

# calculate w
w = np.matmul(linalg.cho_solve(linalg.cho_factor(A), np.eye(2)), b)
print(w)

y_hat_test = np.matmul(np.stack((np.repeat(1.0, len(x_test)), x_test)).T, w)

plt.figure(figsize = (8, 6))
plt.plot(x, y, "k.", markersize = 6)
plt.plot(x_test, y_hat_test, "b-")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


#polynomial regression

def polynomial_regression(x, y, K):
    # calculate A
    A = np.zeros((K + 1, K + 1))
    for i in range(K + 1):
        for j in range(K + 1):
            A[i, j] = np.sum(x**i * x**j)

    # calculate b
    b = np.zeros((K + 1, 1))
    for i in range(K + 1):
        b[i] = np.sum(y * x**i)

    # calculate w
    w = np.matmul(linalg.cho_solve(linalg.cho_factor(A), np.eye(K + 1)), b)

    return(w)


K = 7
w = polynomial_regression(x, y, K)
y_hat_test = np.matmul(np.stack([x_test**k for k in range(K + 1)]).T, w)

plt.figure(figsize = (8, 6))
plt.plot(x, y, "k.", markersize = 6)
plt.plot(x_test, y_hat_test, "b-")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()



#non-linear regression

def nonlinear_regression(x, y, centers, sigma):
    # calculate D
    D = np.vstack((np.ones((1, len(x))), [np.exp(-(x - centers[k])**2 / (2 * sigma**2))
                                          for k in range(len(centers))])).T

    # calculate w
    w = np.matmul(linalg.cho_solve(linalg.cho_factor(np.matmul(D.T, D)), np.eye(len(centers) + 1)),
                  np.matmul(D.T, y[:, None]))

    return(w)

centers = np.linspace(start = 5, stop = 55, num = 11)
sigma = 5
w = nonlinear_regression(x, y, centers, sigma)
D_test = np.vstack((np.ones((1, len(x_test))), [np.exp(-(x_test - centers[k])**2 / (2 * sigma**2))
                                                for k in range(len(centers))])).T
y_hat_test = np.matmul(D_test, w)

plt.figure(figsize = (8, 6))
plt.plot(x, y, "k.", markersize = 6)
plt.plot(centers, np.repeat(0, len(centers)), "r.", markersize = 10)
plt.plot(x_test, y_hat_test, "b-")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()