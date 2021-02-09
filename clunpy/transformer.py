import numpy as np
from sklearn.metrics import confusion_matrix


def confusion_matrix_to_accuracy(confusion_matrix):
    cm = np.array(confusion_matrix)
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def decision_outputs_to_decision_profiles(decision_output_tensor):
    return decision_output_tensor.transpose((1, 0, 2))


def multilabel_predictions_to_decisions(predictions, threshold):
    predictions[predictions >= threshold] = 1
    predictions[predictions < threshold] = 0
    return predictions


def multiclass_predictions_to_decisions(predictions):
    decisions = np.zeros_like(predictions)
    decisions[np.arange(len(decisions)), predictions.argmax(axis=1)] = 1
    return decisions


def decision_outputs_to_configs(decision_outputs):
    configs = decision_outputs[0]
    for i in range(1, len(decision_outputs)):
        configs = np.append(configs, decision_outputs[i], axis=1)
    return configs


def generate_multiclass_confusion_matrices(decision_outputs, true_assignment):
    indexed_ground_truth = np.argmax(true_assignment, axis=1)
    confusion_matrices = np.zeros((np.shape(decision_outputs)[0],
                                   np.shape(true_assignment)[1],
                                   np.shape(true_assignment)[1]), dtype=int)
    for i in range(len(decision_outputs)):
        indexed_do = np.argmax(decision_outputs[i], axis=1)
        confusion_matrices[i] = confusion_matrix(indexed_ground_truth,
                                                 indexed_do,
                                                 np.arange(np.shape(true_assignment)[1]))
    return confusion_matrices
