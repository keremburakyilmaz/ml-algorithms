import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X = np.genfromtxt("fashionmnist_data_points.csv", delimiter = ",") / 255
y = np.genfromtxt("fashionmnist_class_labels.csv", delimiter = ",").astype(int)



i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(i1, cmap = "gray")
plt.show()



# first 60000 data points are included to train
# remaining 10000 data points are included to test
def train_test_split(X, y):
    X_train = X[:60000]
    y_train = y[:60000]
    X_test = X[60000:]
    y_test = y[60000:]
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)

# X = (N, D), W = (D, K), w0 = (1, K)
def sigmoid(X, W, w0):
    scores = np.matmul(X, W) + w0
    probabilities = 1 / (1 + np.exp(-scores))
    return(probabilities)


def one_hot_encoding(y):
    K = np.max(y)
    Y = np.zeros((y.shape[0], K))

    for i in range(Y.shape[0]):
        Y[i, y[i] - 1] = 1

    return(Y)



np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.001, high = 0.001, size = (D, K))
w0_initial = np.random.uniform(low = -0.001, high = 0.001, size = (1, K))



def gradient_W(X, Y_truth, Y_predicted):

    gradient = -np.matmul(X.T, (Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted))

    return(gradient)


def gradient_w0(Y_truth, Y_predicted):

    gradient = -np.sum((Y_truth - Y_predicted) * Y_predicted * (1 - Y_predicted), axis = 0, keepdims=True)

    return(gradient)



def discrimination_by_regression(X_train, Y_train, W_initial, w0_initial):
    eta = 0.15 / X_train.shape[0]
    iteration_count = 250
    count = 0
    objective_values = []

    W = W_initial
    w0 = w0_initial
        
    while True:

        Y_predicted = sigmoid(X_train, W, w0)
        error = 0.5 * np.sum((Y_train - Y_predicted) ** 2)
        objective_values.append(int(error))

        grad_W = gradient_W(X_train, Y_train, Y_predicted)
        grad_w0 = gradient_w0(Y_train, Y_predicted)

        W -= eta * grad_W
        w0 -= eta * grad_w0

        count += 1
        if count == iteration_count:
            break

    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train, W_initial, w0_initial)


def calculate_predicted_class_labels(X, W, w0):

    y_predicted = np.argmax(sigmoid(X, W, w0), axis = 1) + 1

    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)


def calculate_confusion_matrix(y_truth, y_predicted):
    K = np.max(y_truth)
    confusion_matrix = np.zeros((K, K), dtype=int)

    for truth, prediction in zip(y_truth, y_predicted):
        confusion_matrix[truth - 1, prediction - 1] += 1

    confusion_matrix = pd.crosstab(y_predicted, y_truth.T).values

    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)
