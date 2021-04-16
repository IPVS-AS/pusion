import numpy as np
from sklearn.metrics import *

from pusion.util.transformer import multiclass_assignments_to_labels, multilabel_to_multiclass_assignments


def precision(y_true, y_pred):
    """
    Calculate the precision, i.e. TP / (TP + FP).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Precision.
    """
    return precision_score(y_true, y_pred, average='micro')


def recall(y_true, y_pred):
    """
    Calculate the recall, i.e.  TP / (TP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Recall.
    """
    return recall_score(y_true, y_pred, average='micro')


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy, i.e. (TP + TN) / (TP + FP + FN + TN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Accuracy.
    """
    return accuracy_score(y_true, y_pred)


def f1(y_true, y_pred):
    """
    Calculate the F1-score, i.e. 2 * (Precision * Recall) / (Precision + Recall).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: F1-score.
    """
    return f1_score(y_true, y_pred, average='micro')


def jaccard(y_true, y_pred):
    """
    Calculate the Jaccard-score, i.e. TP / (TP + FP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Jaccard-score.
    """
    return jaccard_score(y_true, y_pred, average='micro')


def mean_multilabel_confusion_matrix(y_true, y_pred):
    """
    Calculate the normalized mean confusion matrix across all classes.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: `numpy.array` of shape `(n_classes, n_classes)`. Normalized mean confusion matrix.
    """
    cm_sum = np.sum(multilabel_confusion_matrix(y_true, y_pred, ), axis=0)
    return cm_sum / (len(y_pred) * np.max(cm_sum))


def hamming(y_true, y_pred):
    """
    Calculate the average Hamming Loss.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Average Hamming Loss.
    """
    return hamming_loss(y_true, y_pred)


def log(y_true, y_pred):
    """
    Calculate the Logistic Loss.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Logistic Loss.
    """
    return log_loss(y_true, y_pred)


def cohens_kappa(y1, y2, labels):
    """
    Calculate the Cohen's Kappa annotator agreement score according to :footcite:`cohen1960coefficient`.

    .. footbibliography::

    :param y1: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Labels or class assignments.
    :param y2: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Labels or class assignments.
    :param labels: `list` of all possible labels.
    :return: Cohen's Kappa score.
    """
    cm = confusion_matrix(y1, y2, labels=labels)
    a = np.sum(np.diagonal(cm)) / np.sum(cm)
    e = 0
    for i in range(len(cm)):
        e += np.sum(cm[i, :]) * np.sum(cm[:, i]) / np.sum(cm) ** 2
    if e == 1:
        return 1.0  # case when y1 and y2 are equivalent in their annotation
    return (a - e) / (1 - e)


def pairwise_cohens_kappa_multiclass(decision_tensor):
    """
    Calculate the average of pairwise Cohen's Kappa scores over all multiclass decision outputs.
    E.g., for 3 classifiers `(0,1,2)`, the agreement score is calculated for classifier tuples `(0,1)`, `(0,2)` and
    `(1,2)`. These scores are then averaged over all 3 classifiers.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise (averages) Cohen's Kappa score.
    """
    decision_tensor = np.array(decision_tensor)
    n_classifier = decision_tensor.shape[0]
    n_classes = decision_tensor.shape[2]
    indices = np.array(np.triu_indices(n_classifier, k=1))
    sum_kappa = 0.0
    for i, j in zip(indices[0], indices[1]):
        decision_labels = multiclass_assignments_to_labels([decision_tensor[i], decision_tensor[j]])
        sum_kappa += cohens_kappa(decision_labels[0], decision_labels[1], labels=np.arange(n_classes))
    return sum_kappa / len(indices[0])


def pairwise_cohens_kappa_multilabel(decision_tensor):
    """
    Calculate the average of pairwise Cohen's Kappa scores over all multilabel decision outputs.
    E.g., for 3 classifiers `(0,1,2)`, the agreement score is calculated for classifier tuples `(0,1)`, `(0,2)` and
    `(1,2)`. These scores are then averaged over all 3 classifiers.

    The multilabel outputs are transformed to equivalent multiclass outputs.

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise (averages) Cohen's Kappa score.
    """
    mc_decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
    return pairwise_cohens_kappa_multiclass(mc_decision_tensor)


def __pairwise_micro_score(decision_tensor, score_func):
    """
    A helper function for calculating pairwise score statistics (e.g. Q-statistic).
    """
    decision_tensor = np.array(decision_tensor)
    indices = np.array(np.triu_indices(decision_tensor.shape[0], k=1))
    scores = []
    for i, j in zip(indices[0], indices[1]):
        norm_cm = multilabel_confusion_matrix(decision_tensor[i], decision_tensor[j]) / len(decision_tensor[i])
        mean_cm = np.mean(norm_cm, axis=0)
        a = mean_cm[1, 1]
        b = mean_cm[1, 0]
        c = mean_cm[0, 1]
        d = mean_cm[0, 0]
        scores.append(score_func(a, b, c, d))
    return np.mean(scores)


def q_statistic(decision_tensor):
    """
    Calculate the average of pairwise Q-statistic scores over all decision outputs according to Yule
    :footcite:`udny1900association`.

    .. footbibliography::

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise Q-statistic score.
    """
    def q_stat_func(a, b, c, d): return (a * d - b * c) / (a * d + b * c)
    return __pairwise_micro_score(decision_tensor, q_stat_func)


def correlation(decision_tensor):
    """
    Calculate the average of pairwise correlation scores over all decision outputs according to Polikar
    :footcite:`polikar2006ensemble`.

    .. footbibliography::

    :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise correlation score.
    """
    def correlation_func(a, b, c, d): return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return __pairwise_micro_score(decision_tensor, correlation_func)
