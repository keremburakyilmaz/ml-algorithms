import matplotlib.pyplot as plt
import numpy as np


# read data into memory
data_set_train = np.genfromtxt("data_set_train.csv", delimiter = ",", skip_header = 1)
data_set_test = np.genfromtxt("data_set_test.csv", delimiter = ",", skip_header = 1)

# get x and y values
X_train = data_set_train[:, 0:1]
y_train = data_set_train[:, 1]
X_test = data_set_test[:, 0:1]
y_test = data_set_test[:, 1]

# set drawing parameters
minimum_value = 1.5
maximum_value = 5.1
step_size = 0.001
X_interval = np.arange(start = minimum_value, stop = maximum_value + step_size, step = step_size)
X_interval = X_interval.reshape(len(X_interval), 1)

def plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat):
    fig = plt.figure(figsize = (8, 4))
    plt.plot(X_train[:, 0], y_train, "b.", markersize = 10)
    plt.plot(X_test[:, 0], y_test, "r.", markersize = 10)
    plt.plot(X_interval[:, 0], y_interval_hat, "k-")
    plt.xlabel("Eruption time (min)")
    plt.ylabel("Waiting time to next eruption (min)")
    plt.legend(["training", "test"])
    plt.show()
    return(fig)


def decision_tree_regression_train(X_train, y_train, P):
    # create necessary data structures
    node_indices = {0: np.arange(X_train.shape[0])}
    is_terminal = {}
    need_split = {0: True}

    node_features = {}
    node_splits = {}
    node_means = {}
    
    while any(need_split.values()):
        for node_id in [key for key, val in need_split.items() if val]:
            data_indices = node_indices[node_id]
            X_node = X_train[data_indices]
            y_node = y_train[data_indices]

            node_means[node_id] = np.mean(y_node)
            if len(data_indices) <= P:
                is_terminal[node_id] = True
                need_split[node_id] = False
            else:
                is_terminal[node_id] = False
                need_split[node_id] = False

                best_feature = 0
                unique_values = np.unique(X_node[:, best_feature])
                split_positions = (unique_values[: -1] + unique_values[1: ]) / 2

                best_split = None
                best_error = float("inf")

                for split in split_positions:
                    left_indices = data_indices[X_node[:, best_feature] <= split]                    
                    right_indices = data_indices[X_node[:, best_feature] > split]

                    left_mean = np.mean(y_train[left_indices])
                    right_mean = np.mean(y_train[right_indices])

                    error = np.sum((y_train[left_indices] - left_mean) ** 2) + np.sum((y_train[right_indices] - right_mean) ** 2)

                    if error < best_error:
                        best_error = error
                        best_split = split
                    
                node_features[node_id] = best_feature
                node_splits[node_id] = best_split

                left_id = 2 * node_id + 1
                right_id = 2 * node_id + 2

                node_indices[left_id] = data_indices[X_node[:, best_feature] <= best_split]
                node_indices[right_id] = data_indices[X_node[:, best_feature] > best_split]

                need_split[left_id] = True
                need_split[right_id] = True

    return(is_terminal, node_features, node_splits, node_means)



def decision_tree_regression_test(X_query, is_terminal, node_features, node_splits, node_means):
    y_hat = np.zeros(X_query.shape[0])

    for i, x in enumerate(X_query):
        node_id = 0
        while not is_terminal[node_id]:
            feature = node_features[node_id]
            split = node_splits[node_id]
            if x[feature] <= split:
                node_id = 2 * node_id + 1
            else:
                node_id = 2 * node_id + 2

        y_hat[i] = node_means[node_id]

    return(y_hat)



def traverse(node_id, conditions, rules):
    if is_terminal[node_id]:
        rules.append((conditions, node_means[node_id]))
    else:
        feature = node_features[node_id]
        split = node_splits[node_id]

        traverse(2 * node_id + 1, conditions + [f"x{feature + 1} <= {split:.2f}"], rules)
        traverse(2 * node_id + 2, conditions + [f"x{feature + 1} > {split:.2f}"], rules)



def extract_rule_sets():
    rules = []
    traverse(0, [], rules)

    for i, (conds, mean) in enumerate(rules):
        print(f"Node {i + 1:02}: {conds} => {mean:.2f}")


P = 20
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

P = 50
is_terminal, node_features, node_splits, node_means = decision_tree_regression_train(X_train, y_train, P)
y_interval_hat = decision_tree_regression_test(X_interval, is_terminal, node_features, node_splits, node_means)
fig = plot_figure(X_train, y_train, X_test, y_test, X_interval, y_interval_hat)

y_train_hat = decision_tree_regression_test(X_train, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_train - y_train_hat)**2))
print("RMSE on training set is {} when P is {}".format(rmse, P))

y_test_hat = decision_tree_regression_test(X_test, is_terminal, node_features, node_splits, node_means)
rmse = np.sqrt(np.mean((y_test - y_test_hat)**2))
print("RMSE on test set is {} when P is {}".format(rmse, P))

extract_rule_sets()
