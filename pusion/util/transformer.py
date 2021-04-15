import math

import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix


def confusion_matrices_to_accuracy_vector(confusion_matrix_tensor):
    """
    Convert confusion matrices of respective classification to an accuracy vector.

    :param confusion_matrix_tensor: `numpy.array` of shape `(n_classifier, n_classes, n_classes)`
            Confusion matrices.
    :return: One-dimensional `numpy.array` of shape of length `n_classifier` containing the accuracy for each confusion
            matrix.
    """
    accuracies = np.zeros(len(confusion_matrix_tensor))
    for i in range(len(confusion_matrix_tensor)):
        accuracies[i] = confusion_matrix_to_accuracy(confusion_matrix_tensor[i])
    return accuracies


def confusion_matrix_to_accuracy(cm):
    """
    Calculate the accuracy out of the given confusion matrix.

    :param cm: `numpy.array` of shape `(n_classes, n_classes)`.
            Confusion matrix.
    :return: The accuracy.
    """
    cm = np.array(cm)
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def multilabel_cr_confusion_matrices_to_avg_accuracy(label_cms):
    """
    Calculate the average accuracy for the given confusion matrices generated from complementary redundant multilabel
    output.

    :param label_cms: `list` of confusion matrices given as 2-dimensional `numpy.array` respectively.
    :return: The average accuracy.
    """
    avg = 0.0
    for label_cm in label_cms:
        avg += confusion_matrix_to_accuracy(label_cm)
    return avg / len(label_cms)


def decision_tensor_to_decision_profiles(decision_tensor):
    """
    Transform the given decision tensor to decision profiles for each respective sample.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of either crisp or continuous decision outputs by different classifiers per sample.
    :return: `numpy.array` of shape `(n_samples, n_classifier, n_classes)`.
            Decision profiles.
    """
    return np.array(decision_tensor).transpose((1, 0, 2))


def multilabel_predictions_to_decisions(predictions, threshold=0.5):
    """
    Transform a continuously valued tensor of multilabel decisions to crisp decision outputs.

    :param predictions: `numpy.array` of any shape. Continuous predictions.
    :param threshold: `float`. A threshold value, based on which the crisp output is constructed.
    :return: `numpy.array` of the same shape as ``predictions``. Crisp decision outputs.
    """
    return (predictions >= threshold) * np.ones_like(predictions)


def multiclass_predictions_to_decisions(predictions):
    """
    Transform a continuously valued matrix of multiclass decisions to crisp decision outputs.

    :param predictions: `numpy.array` of shape `(n_classifier, n_classes)`. Continuous predictions.
    :return: `numpy.array` of the same shape as ``predictions``. Crisp decision outputs.
    """
    decisions = np.zeros_like(predictions)
    decisions[np.arange(len(decisions)), predictions.argmax(axis=1)] = 1
    return decisions


def decision_tensor_to_configs(decision_outputs):
    """
    Transform decision outputs to decision configs. A decision config shows concatenated classification outputs of each
    classifier per sample.

    :param decision_outputs: `numpy.array` of shape `(n_classifier, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :return: `numpy.array` of shape `(n_samples, n_classes*)`, `n_classes*` is the sum of all classes covered by
            all classifiers.
    """
    return np.concatenate(decision_outputs, axis=1)


def multiclass_assignments_to_labels(assignments):
    """
    Transform multiclass assignments to labels. A matrix of shape `(n_samples, n_classes)` is converted to a vector
    of shape `(n_samples,)`, with element-wise labels represented in integers from `0` to `n_classes - 1`.

    :param assignments: `numpy.array` of shape `(n_samples, n_classes)`. Multiclass assignments.
    :return: `numpy.array` of shape `(n_samples,)` with an integer label per element.
    """
    assignments = np.array(assignments)
    return np.argmax(assignments, axis=assignments.ndim - 1)


def transform_labels_to_class_assignments(labels, n_classes):
    """
    Transform labels to multiclass assignments. A vector of shape `(n_samples,)`, with element-wise labels is converted
    to the assignment matrix of shape `(n_samples, n_classes)`.

    :param labels: `numpy.array` of shape `(n_samples,)` with an integer label per element.
    :param n_classes: Number of classes to be considered.
    :return: `numpy.array` of shape `(n_samples, n_classes)`. Multiclass assignments.
    """
    assignments = np.zeros((len(labels), n_classes))
    assignments[np.arange(len(labels)), labels] = 1
    return assignments


def transform_label_tensor_to_class_assignment_tensor(label_tensor, n_classes):
    """
    Transform a tensor label tensor of shape `(n_classifier, n_samples)` to the tensor of class assignments of shape
    `(n_classifier, n_samples, n_classes)`. A label is an integer between `0` and `n_classes - 1`.

    :param label_tensor: `numpy.array` of shape `(n_classifier, n_samples)`. Label tensor.
    :param n_classes: Number of classes to be considered.
    :return: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`. Class assignment tensor (decision tensor).
    """
    label_tensor = np.array(label_tensor)
    assignments = np.zeros((label_tensor.shape[0], label_tensor.shape[1], n_classes))
    for i in range(label_tensor.shape[0]):
        assignments[i, np.arange(label_tensor.shape[1]), label_tensor[i]] = 1
    return assignments


def transform_label_vector_to_class_assignment_matrix(label_vector, n_classes):
    """
    Transform labels to multiclass assignments. A vector of shape `(n_samples,)`, with element-wise labels is converted
    to the assignment matrix of shape `(n_samples, n_classes)`.

    :param label_vector: `numpy.array` of shape `(n_samples,)` with an integer label per element.
    :param n_classes: Number of classes to be considered.
    :return: `numpy.array` of shape `(n_samples, n_classes)`. Multiclass assignments.
    """
    if label_vector.ndim > 1:  # TODO: label vector is always one-dimensional
        return label_vector
    assignments = transform_label_tensor_to_class_assignment_tensor(np.array([label_vector]), n_classes)
    return assignments.squeeze()


def generate_multiclass_confusion_matrices(decision_tensor, true_assignment):  # TODO move to generator
    """
    Generate multiclass confusion matrices out of the given decision tensor and true assignments.
    Continuous outputs are converted to multiclass assignments using the MAX rule.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.
    :param true_assignment: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments which are considered true for calculating confusion matrices.
    :return: `numpy.array` of shape `(n_classifier, n_samples, n_samples)`. Confusion matrices per classifier.
    """
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
    """
    Generate multilabel confusion matrices for complementary-redundant multilabel classification outputs.

    :param decision_outputs: `numpy.array` of shape `(n_classifier, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :param true_assignment: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments which are considered true for calculating confusion matrices.
    :param coverage: `list` of `list` elements. Each list contains classes as integers covered by a classifier, which
            is identified by the positional index of the respective list.
    :return: List of multilabel confusion matrices.
    """
    cr_confusion_matrices = []
    for i, do in enumerate(decision_outputs):
        ta = intercept_normal_class(true_assignment[:, coverage[i]], override=True)
        cms = multilabel_confusion_matrix(y_true=ta, y_pred=do, labels=np.arange(len(coverage[i])))
        cr_confusion_matrices.append(cms)
    return cr_confusion_matrices


def multilabel_to_multiclass_assignments(decision_tensor):
    """
    Transform the multilabel decision tensor to the equivalent multiclass decision tensor using the power set method.
    The multilabel class assignments are considered as a binary number which represents a new class in the
    multiclass decision space. E.g. the assignment to the classes `0` and `2` (`[1,0,1]`) is converted to the class `5`,
    which is one of the `2^3` classes in the multiclass decision space.
    This method is inverse to the ``multiclass_to_multilabel_assignments`` method.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multilabel decision outputs by different classifiers per sample.
    :return: `numpy.array` of shape `(n_classifier, n_samples, 2^n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    """
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
    """
    Transform the multiclass decision tensor to the equivalent multilabel decision tensor using the inverse
    power set method. The multiclass assignment is considered as a decimal which is converted to a binary number, which
    in turn represents the multilabel class assignment. E.g. the class assignment to the class `3` `([0,0,0,1])` is
    converted to the multilabel class assignment `[1,1]` (classes `0` and `1` in the multilabel decision space).
    This method is inverse to the ``multilabel_to_multiclass_assignments`` method.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multilabel decision outputs by different classifiers per sample.
    :return: `numpy.array` of shape `(n_classifier, n_samples, log_2(n_classes))`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    """
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
    """
    Convert `list` decision outputs to `numpy.array` decision tensor, if possible.
    """
    try:
        return np.array(decision_outputs, dtype=float)
    except ValueError:
        return decision_outputs


def intercept_normal_class(y, override=False):
    """
    Intercept the normal class for the given decision matrix, i.e. a normal class is assigned to each zero vector
    class assignment. E.g. the assignment `[0,0,0,0]` is transformed to [1,0,0,0]`, under the assumption that `0`
    is a normal class.

    :param y: `numpy.array` of shape `(n_samples, n_classes)`. Matrix of decision outputs.
    :param override: If `true`, the class `0` is assumed as a normal class. Otherwise a new class is prepended to
            existing classes.
    :return: `numpy.array` of shape `(n_samples, n_classes)` for `override=False`.
            `numpy.array` of shape `(n_samples, n_classes + 1)` for `override=True`.
            Matrix of decision outputs with intercepted normal class.
    """
    y = np.array(y)
    if not override:
        y = np.column_stack((np.zeros(y.shape[0], dtype=int), y))
    z_indices = np.where(~y.any(axis=1))[0]
    # set the normal class to 1 for zero vectors
    y[z_indices, 0] = 1
    return y


def intercept_normal_class_in_tensor(decision_tensor, override=False):
    """
    Intercept the normal class for the given decision matrix, i.e. a normal class is assigned to each zero vector
    class assignment. E.g. the assignment `[0,0,0,0]` is transformed to [1,0,0,0]`, under the assumption that `0`
    is a normal class.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of decision outputs by different classifiers per sample.
    :param override: If `true`, the class `0` is assumed as a normal class. Otherwise a new class is prepended to
            existing classes.
    :return: `numpy.array` of shape `(n_samples, n_classes)` for `override=False`.
            `numpy.array` of shape `(n_samples, n_classes + 1)` for `override=True`.
            Matrix of decision outputs with intercepted normal class.
        """
    normalized_decision_outputs = []
    for i in range(len(decision_tensor)):
        normalized_decision_outputs.append(intercept_normal_class(decision_tensor[i], override))
    return decision_outputs_to_decision_tensor(normalized_decision_outputs)
