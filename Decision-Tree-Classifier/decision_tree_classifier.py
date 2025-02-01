import numpy as np
import pandas as pd

def safelog2(x):
    if x == 0:
        return(0)
    else:
        return(np.log2(x))
    

# read data into memory
data_set = np.genfromtxt("data_set.csv", delimiter = ",")

# get number of samples and number of features
N = data_set.shape[0]
D = data_set.shape[1] - 1

# get X and y values
X = data_set[:,0:D]
y = data_set[:,D:(D + 1)].astype(int)

# get number of classes
K = np.max(y)

# get train and test splits
train_indices = np.concatenate((np.arange(0, 25),
                                np.arange(50, 75),
                                np.arange(100, 125)))
test_indices = np.setdiff1d(range(N), train_indices)

X_train = X[train_indices,:]
y_train = y[train_indices]
X_test = X[test_indices,:]
y_test = y[test_indices]

# get numbers of train and test samples
N_train = len(y_train)
N_test = len(y_test)


# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}

node_features = {}
node_splits = {}
node_frequencies = {}

# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True


# learning algorithm
while True:
    # find nodes that need splitting
    split_nodes = [key for key, value in need_split.items()
                   if value == True]
    
    # check whether we reach all terminal nodes
    if len(split_nodes) == 0:
        break

    # find best split positions for all nodes
    for split_node in split_nodes:
        data_indices = node_indices[split_node]
        need_split[split_node] = False
        node_frequencies[split_node] = [np.sum(y_train[data_indices] == (c + 1))
                                        for c in range(K)]
        if len(np.unique(y_train[data_indices])) == 1:
            is_terminal[split_node] = True
        else:
            is_terminal[split_node] = False

            best_scores = np.repeat(0.0, D)
            best_splits = np.repeat(0.0, D)
            for d in range(D):
                unique_values = np.sort(np.unique(X_train[data_indices, d]))
                split_positions = (unique_values[1:] + unique_values[:-1]) / 2
                split_scores = np.repeat(0.0, len(split_positions))
                for s in range(len(split_positions)):
                    left_indices = data_indices[X_train[data_indices, d] > \
                                   split_positions[s]]
                    right_indices = data_indices[X_train[data_indices, d] <= \
                                    split_positions[s]]
                    split_scores[s] = -(len(left_indices) / len(data_indices) * \
                                        np.sum([np.mean(y_train[left_indices] == (c + 1)) * \
                                                safelog2(np.mean(y_train[left_indices] == (c + 1)))
                                                for c in range(K)]) + \
                                        len(right_indices) / len(data_indices) * \
                                        np.sum([np.mean(y_train[right_indices] == (c + 1)) * \
                                                safelog2(np.mean(y_train[right_indices] == (c + 1)))
                                                for c in range(K)]))
                best_scores[d] = np.min(split_scores)
                best_splits[d] = split_positions[np.argmin(split_scores)]

            # decide where to split on which feature
            split_d = np.argmin(best_scores)

            node_features[split_node] = split_d
            node_splits[split_node] = best_splits[split_d]
            
            # create left node using the selected split
            left_indices = data_indices[X_train[data_indices, split_d] > \
                           best_splits[split_d]]
            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True
      
            # create right node using the selected split
            right_indices = data_indices[X_train[data_indices, split_d] <= \
                            best_splits[split_d]]
            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True


# extract rules
terminal_nodes = [key for key, value in is_terminal.items()
                  if value == True]
for terminal_node in terminal_nodes:
    index = terminal_node
    rules = np.array([])
    while index > 1:
        parent = np.floor(index / 2)
        if index % 2 == 0:
            # if node is left child of its parent
            rules = np.append(rules, 
                              "x{:d} > {:.2f}".format(node_features[parent] + 1,
                                                      node_splits[parent]))
        else:
            # if node is right child of its parent
            rules = np.append(rules,
                              "x{:d} <= {:.2f}".format(node_features[parent] + 1,
                                                       node_splits[parent]))
        index = parent
    rules = np.flip(rules)
    print("Node {:02}: {} => {}".format(terminal_node, rules, node_frequencies[terminal_node]))


# traverse tree for training data points
y_predicted = np.repeat(0, N_train)
for i in range(N_train):
    index = 1
    while True:
        if is_terminal[index] == True:
            y_predicted[i] = np.argmax(node_frequencies[index]) + 1
            break
        else:
            if X_train[i, node_features[index]] > node_splits[index]:
                index = 2 * index
            else:
                index = 2 * index + 1
confusion_matrix = pd.crosstab(y_predicted, y_train.T,
                               rownames = ["y_predicted"],
                               colnames = ["y_train"])
print(confusion_matrix)


# traverse tree for test data points
y_predicted = np.repeat(0, N_test)
for i in range(N_test):
    index = 1
    while True:
        if is_terminal[index] == True:
            y_predicted[i] = np.argmax(node_frequencies[index]) + 1
            break
        else:
            if X_test[i, node_features[index]] > node_splits[index]:
                index = 2 * index
            else:
                index = 2 * index + 1
confusion_matrix = pd.crosstab(y_predicted, y_test.T,
                               rownames = ["y_predicted"],
                               colnames = ["y_test"])
print(confusion_matrix)