import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

data_set = np.genfromtxt("dataset.csv", delimiter = ",")

X = data_set[:, [0, 1]]

K = 4

def initialize_parameters(X, K):
    initial_centroids = np.genfromtxt("initial_centroids.csv", delimiter=",")

    assignments = np.argmin(np.linalg.norm(X[:, None] - initial_centroids[None, :], axis=2), axis=1)

    means = np.array([X[assignments == k].mean(axis=0) for k in range(K)])

    covariances = np.array([np.cov(X[assignments == k].T) if len(X[assignments == k]) > 1 else np.eye(X.shape[1])
                            for k in range(K)])
    
    priors = np.array([np.mean(assignments == k) for k in range(K)])

    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)


def em_clustering_algorithm(X, K, means, covariances, priors):
    N = X.shape[0]

    for iteration in range(100):
        responsibilities = np.zeros((N, K))
        for k in range(K):
            responsibilities[:, k] = priors[k] * stats.multivariate_normal.pdf(X, means[k], covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        Nk = responsibilities.sum(axis=0)
        priors = Nk / N
        means = (responsibilities.T @ X) / Nk[:, None]
        covariances = []
        for k in range(K):
            diff = X - means[k]
            covariances.append((responsibilities[:, k][:, None] * diff).T @ diff / Nk[k])
        covariances = np.array(covariances)

    assignments = np.argmax(responsibilities, axis=1)
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)


def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    plt.figure(figsize=(10, 10))

    colors = ['r', 'g', 'b', 'm']
    for k in range(K):
        plt.scatter(X[assignments == k, 0], X[assignments == k, 1], color=colors[k], alpha=0.5, label=f"Cluster {k+1}")

    for k in range(K):
        eigvals, eigvecs = linalg.eigh(group_covariances[k])
        theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals * stats.chi2.ppf(0.99, 2))
        ell = plt.matplotlib.patches.Ellipse(xy=group_means[k], width=width, height=height, angle=theta, edgecolor=colors[k], linestyle='--', fill=False)
        plt.gca().add_patch(ell)

    for k in range(K):
        eigvals, eigvecs = linalg.eigh(covariances[k])
        theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals * stats.chi2.ppf(0.99, 2))
        ell = plt.matplotlib.patches.Ellipse(xy=means[k], width=width, height=height, angle=theta, edgecolor=colors[k], linestyle='-', fill=False)
        plt.gca().add_patch(ell)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("EM Clustering Results")
    plt.grid()
    plt.show()
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)