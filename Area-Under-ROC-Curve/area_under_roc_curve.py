import matplotlib.pyplot as plt
import numpy as np

true_labels = np.genfromtxt("true_labels.csv", delimiter = ",", dtype = "int")
predicted_probabilities1 = np.genfromtxt("predicted_probabilities1.csv", delimiter = ",")
predicted_probabilities2 = np.genfromtxt("predicted_probabilities2.csv", delimiter = ",")


def calculate_threholds(predicted_probabilities):
    predicted_probabilities = np.concatenate(([0], np.sort(predicted_probabilities), [1]))
    thresholds = [(predicted_probabilities[i] + predicted_probabilities[i+1]) / 2 for i in range(len(predicted_probabilities) - 1)]
    return thresholds


thresholds1 = calculate_threholds(predicted_probabilities1)

thresholds2 = calculate_threholds(predicted_probabilities2)

def calculate_fp_and_tp_rates(true_labels, predicted_probabilities, thresholds):
    fp_rates = []
    tp_rates = []

    for threshold in thresholds:
        predictions = np.zeros(len(true_labels))
        for i in range(len(predictions)):
            predictions[i] = 1 if predicted_probabilities[i] > threshold else -1

        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == -1))
        tn = np.sum((predictions == -1) & (true_labels == -1))
        fn = np.sum((predictions == -1) & (true_labels == 1))

        tp_rate = (tp / (tp + fn))
        fp_rate = (fp / (fp + tn))

        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)
    return fp_rates, tp_rates


fp_rates1, tp_rates1 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities1, thresholds1)

fp_rates2, tp_rates2 = calculate_fp_and_tp_rates(true_labels, predicted_probabilities2, thresholds2)

fig = plt.figure(figsize = (5, 5))
plt.plot(fp_rates1, tp_rates1, label = "Classifier 1")
plt.plot(fp_rates2, tp_rates2, label = "Classifier 2")
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.show()
fig.savefig("roc_curves.pdf", bbox_inches = "tight")


def calculate_auroc(fp_rates, tp_rates):
    auroc = 0
    for i in range(1, len(fp_rates)):
        width = fp_rates[i] - fp_rates[i - 1]
        height = (tp_rates[i] + tp_rates[i - 1])
        auroc += - width * height / 2
    return auroc

auroc1 = calculate_auroc(fp_rates1, tp_rates1)
print("The area under the ROC curve for Algorithm 1 is {}.".format(auroc1))
auroc2 = calculate_auroc(fp_rates2, tp_rates2)
print("The area under the ROC curve for Algorithm 2 is {}.".format(auroc2))