import math

import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


def confusion_matrices_to_accuracy_vector(confusion_matrix_tensor):
    accuracies = np.zeros(len(confusion_matrix_tensor))
    for i in range(len(confusion_matrix_tensor)):
        accuracies[i] = confusion_matrix_to_accuracy(confusion_matrix_tensor[i])
    return accuracies


def confusion_matrix_to_accuracy(cm):
    cm = np.array(cm)
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def multilabel_cr_confusion_matrices_to_avg_accuracy(label_cms):
    avg = 0.0
    for label_cm in label_cms:
        avg += confusion_matrix_to_accuracy(label_cm)
    return avg / len(label_cms)


def decision_tensor_to_decision_profiles(decision_output_tensor):
    return np.array(decision_output_tensor).transpose((1, 0, 2))


def multilabel_predictions_to_decisions(predictions, threshold=0.5):
    return (predictions >= threshold) * np.ones_like(predictions)


def multiclass_predictions_to_decisions(predictions):
    decisions = np.zeros_like(predictions)
    decisions[np.arange(len(decisions)), predictions.argmax(axis=1)] = 1
    return decisions


def decision_tensor_to_configs(decision_outputs):
    return np.concatenate(decision_outputs, axis=1)


def multiclass_assignments_to_labels(assignments):
    assignments = np.array(assignments)
    return np.argmax(assignments, axis=assignments.ndim - 1)


def transform_labels_to_class_assignments(labels, n_classes):
    assignments = np.zeros((len(labels), n_classes))
    assignments[np.arange(len(labels)), labels] = 1
    return assignments


def transform_label_tensor_to_class_assignment_tensor(label_tensor, n_classes):
    label_tensor = np.array(label_tensor)
    assignments = np.zeros((label_tensor.shape[0], label_tensor.shape[1], n_classes))
    for i in range(label_tensor.shape[0]):
        assignments[i, np.arange(label_tensor.shape[1]), label_tensor[i]] = 1
    return assignments


def transform_label_vector_to_class_assignment_matrix(label_vector, n_classes):
    if label_vector.ndim > 1:  # TODO: label vector is always one-dimensional
        return label_vector
    assignments = transform_label_tensor_to_class_assignment_tensor(np.array([label_vector]), n_classes)
    return assignments.squeeze()


def generate_multiclass_confusion_matrices(decision_tensor, true_assignment):  # TODO move to generator
    true_assignment_labels = np.argmax(true_assignment, axis=1)
    confusion_matrices = np.zeros((np.shape(decision_tensor)[0],
                                   np.shape(true_assignment)[1],
                                   np.shape(true_assignment)[1]), dtype=int)
    for i in range(len(decision_tensor)):
        decision_tensor_labels = np.argmax(decision_tensor[i], axis=1)
        confusion_matrices[i] = confusion_matrix(y_true=true_assignment_labels,
                                                 y_pred=decision_tensor_labels,
                                                 labels=np.arange(np.shape(true_assignment)[1]))
    return confusion_matrices


def generate_multilabel_cr_confusion_matrices(decision_outputs, true_assignment, coverage):  # TODO reverse args
    cr_confusion_matrices = []
    for i, do in enumerate(decision_outputs):
        ta = intercept_normal_class(true_assignment[:, coverage[i]], override=True)
        cms = multilabel_confusion_matrix(y_true=ta, y_pred=do, labels=np.arange(len(coverage[i])))
        cr_confusion_matrices.append(cms)
    return cr_confusion_matrices


def multilabel_to_multiclass_assignments(decision_tensor):
    decision_tensor = np.array(decision_tensor)
    input_is_matrix = False
    if decision_tensor.ndim == 2:
        decision_tensor = decision_tensor[np.newaxis, :]
        input_is_matrix = True
    mc_decision_tensor = np.zeros((decision_tensor.shape[0],
                                   decision_tensor.shape[1],
                                   2 ** decision_tensor.shape[2]), dtype=int)
    for i in range(decision_tensor.shape[0]):
        for j in range(decision_tensor.shape[1]):
            new_class_index = sum(b << i for i, b in enumerate(decision_tensor[i, j]))
            mc_decision_tensor[i, j, new_class_index] = 1
    if input_is_matrix:
        return mc_decision_tensor.squeeze()
    return mc_decision_tensor


def multiclass_to_multilabel_assignments(decision_tensor):
    decision_tensor = np.array(decision_tensor)
    input_is_matrix = False
    if decision_tensor.ndim == 2:
        decision_tensor = decision_tensor[np.newaxis, :]
        input_is_matrix = True
    ml_decision_tensor = np.zeros((decision_tensor.shape[0],
                                   decision_tensor.shape[1],
                                   int(math.log(decision_tensor.shape[2], 2))), dtype=int)
    for i in range(decision_tensor.shape[0]):
        for j in range(decision_tensor.shape[1]):
            class_indices = np.where(decision_tensor[i, j])[0]
            if class_indices.size > 0:
                ml_decision = np.flip(np.array([int(x) for x in bin(class_indices[0])[2:]]))
                ml_decision_tensor[i, j, range(len(ml_decision))] = ml_decision
    if input_is_matrix:
        return ml_decision_tensor.squeeze()
    return ml_decision_tensor


# TODO check further usability
def decision_outputs_to_decision_tensor(decision_outputs):
    try:
        return np.array(decision_outputs, dtype=float)
    except ValueError:
        return decision_outputs


def intercept_normal_class(y, override=False):
    y = np.array(y)
    if not override:
        y = np.column_stack((np.zeros(y.shape[0], dtype=int), y))
    z_indices = np.where(~y.any(axis=1))[0]
    # set the normal class to 1 for zero vectors
    y[z_indices, 0] = 1
    return y


def intercept_normal_class_in_tensor(decision_tensor, override=False):
    normalized_decision_outputs = []
    for i in range(len(decision_tensor)):
        normalized_decision_outputs.append(intercept_normal_class(decision_tensor[i], override))
    return decision_outputs_to_decision_tensor(normalized_decision_outputs)
