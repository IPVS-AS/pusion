import numpy as np
from pusion.util.constants import Problem
from sklearn.metrics import *

from pusion.auto.detector import determine_problem
from pusion.util.transformer import multiclass_assignments_to_labels, multilabel_to_multiclass_assignments


def micro_precision(y_true, y_pred):
    """
    Calculate the micro precision, i.e. TP / (TP + FP).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The micro precision.
    """
    return precision_score(y_true, y_pred, average='micro')


def micro_recall(y_true, y_pred):
    """
    Calculate the micro recall, i.e.  TP / (TP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The micro recall.
    """
    return recall_score(y_true, y_pred, average='micro')


def micro_f1(y_true, y_pred):
    """
    Calculate the micro F1-score, i.e. 2 * (Precision * Recall) / (Precision + Recall).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The micro F1-score.
    """
    return f1_score(y_true, y_pred, average='micro')


def micro_f2(y_true, y_pred):
    """
    Calculate the micro F2-score (beta=2).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The micro F2-score.
    """
    return fbeta_score(y_true, y_pred, average='micro', beta=2)


def micro_jaccard(y_true, y_pred):
    """
    Calculate the micro Jaccard-score, i.e. TP / (TP + FP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The micro Jaccard-score.
    """
    return jaccard_score(y_true, y_pred, average='micro')


def macro_precision(y_true, y_pred):
    """
    Calculate the macro precision, i.e. TP / (TP + FP).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The macro precision.
    """
    return precision_score(y_true, y_pred, average='macro')


def macro_recall(y_true, y_pred):
    """
    Calculate the macro recall, i.e.  TP / (TP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The macro recall.
    """
    return recall_score(y_true, y_pred, average='macro')


def macro_f1(y_true, y_pred):
    """
    Calculate the macro F1-score, i.e. 2 * (Precision * Recall) / (Precision + Recall).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The macro F1-score.
    """
    return f1_score(y_true, y_pred, average='macro')


def macro_f2(y_true, y_pred):
    """
    Calculate the macro F2-score (beta=2).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The macro F2-score.
    """
    return fbeta_score(y_true, y_pred, average='macro', beta=2)


def macro_jaccard(y_true, y_pred):
    """
    Calculate the macro Jaccard-score, i.e. TP / (TP + FP + FN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The macro Jaccard-score.
    """
    return jaccard_score(y_true, y_pred, average='macro')


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy, i.e. (TP + TN) / (TP + FP + FN + TN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Accuracy.
    """
    return accuracy_score(y_true, y_pred)


def balanced_multiclass_accuracy(y_true, y_pred):
    """
    Calculate the balanced accuracy, i.e. (Precision + Recall) / 2.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Accuracy.
    """
    if y_true.ndim > 1 or y_pred.ndim > 1:
        y_true = multiclass_assignments_to_labels(y_true)
        y_pred = multiclass_assignments_to_labels(y_pred)
    return balanced_accuracy_score(y_true, y_pred)


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


def mean_confidence(y_true, y_pred):
    """
    Calculate the mean confidence for continuous multiclass and multilabel classification outputs.

    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted class assignments.

    :return: Mean confidence.
    """
    return 1 - np.sum(np.abs(y_true - y_pred)) / (y_true.shape[0] * y_true.shape[1])


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


def pairwise_cohens_kappa(decision_tensor):
    """
    Calculate the average of pairwise Cohen's Kappa scores over all multiclass decision outputs.
    E.g., for 3 classifiers `(0,1,2)`, the agreement score is calculated for classifier tuples `(0,1)`, `(0,2)` and
    `(1,2)`. These scores are then averaged over all 3 classifiers.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise (averages) Cohen's Kappa score.
    """
    decision_tensor = np.array(decision_tensor)

    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)

    n_classifiers = decision_tensor.shape[0]
    n_classes = decision_tensor.shape[2]
    indices = np.array(np.triu_indices(n_classifiers, k=1))
    sum_kappa = 0.0
    for i, j in zip(indices[0], indices[1]):
        decision_labels = multiclass_assignments_to_labels([decision_tensor[i], decision_tensor[j]])
        sum_kappa += cohens_kappa(decision_labels[0], decision_labels[1], labels=np.arange(n_classes))
    return sum_kappa / len(indices[0])


def __relations(y1, y2, y_true):
    """
    A helper function for calculating the correctness relations between two classifier outputs.
    `A` accumulates samples which are correctly classified by both classifiers, `B` accumulates those which are
    correctly classified by `c_1` but not by `c_2` and so on.
    """
    n_samples = len(y_true)
    a, b, c, d = 0, 0, 0, 0
    for i in range(n_samples):
        if np.all(y1[i] == y_true[i]) and np.all(y_true[i] == y2[i]):
            a += 1  # both classifiers are correct.
        elif np.all(y1[i] == y_true[i]) and np.any(y_true[i] != y2[i]):
            b += 1  # c1 is correct, c2 is wrong.
        elif np.any(y1[i] != y_true[i]) and np.all(y_true[i] == y2[i]):
            c += 1  # c1 is wrong, c2 is correct.
        elif np.any(y1[i] != y_true[i]) and np.any(y_true[i] != y2[i]):
            d += 1  # both classifiers are wrong.
    return a/n_samples, b/n_samples, c/n_samples, d/n_samples


def __pairwise_avg_score(decision_tensor, true_assignments, score_func):
    """
    A helper function for calculating pairwise average score statistics.
    """
    decision_tensor = np.array(decision_tensor)
    indices = np.array(np.triu_indices(decision_tensor.shape[0], k=1))
    scores = []
    for i, j in zip(indices[0], indices[1]):
        scores.append(score_func(decision_tensor[i], decision_tensor[j], true_assignments))
    return np.mean(scores)


def correlation(y1, y2, y_true):
    """
    Calculate the correlation score for decision outputs of two classifiers according to Kuncheva
    :footcite:`kuncheva2014combining`.

    .. footbibliography::

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Correlation score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))


def q_statistic(y1, y2, y_true):
    """
    Calculate the Q statistic score for decision outputs of two classifiers according to Yule
    :footcite:`udny1900association`.

    .. footbibliography::

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Correlation score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return (a * d - b * c) / (a * d + b * c)


def kappa_statistic(y1, y2, y_true):
    """
    Calculate the kappa score for decision outputs of two classifiers according to Kuncheva
    :footcite:`kuncheva2014combining`.

    .. footbibliography::

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Kappa score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return (2 * (a * d - b * c))/((a + b)*(b + d) + (a + c)*(c + d))


def disagreement(y1, y2, y_true):
    """
    Calculate the disagreement for decision outputs of two classifiers, i.e. the percentage of samples which are
    correctly classified by exactly one of the classifiers.

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Disagreement score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return b + c


def double_fault(y1, y2, y_true):
    """
    Calculate the double fault for decision outputs of two classifiers, i.e. the percentage of samples which are
    misclassified by both classifiers.

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Double fault score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return d


def abs_correlation(y1, y2, y_true):
    """
    Calculate the absolute correlation score for decision outputs of two classifiers.

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Correlation score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return np.abs((a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d)))


def abs_q_statistic(y1, y2, y_true):
    """
    Calculate the absolute Q statistic score for decision outputs of two classifiers.

    :param y1: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the first classifier.
    :param y2: `numpy.array` of shape `(n_samples, n_classes)`.
            Crisp multiclass decision outputs by the second classifier.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Correlation score.
    """
    a, b, c, d = __relations(y1, y2, y_true)
    return np.abs((a * d - b * c) / (a * d + b * c))


def pairwise_correlation(decision_tensor, true_assignments):
    """
    Calculate the average of the pairwise absolute correlation scores over all decision outputs.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Pairwise correlation score.
    """
    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
        true_assignments = multilabel_to_multiclass_assignments(true_assignments)
    return __pairwise_avg_score(decision_tensor, true_assignments, abs_correlation)


def pairwise_q_statistic(decision_tensor, true_assignments):
    """
    Calculate the average of the pairwise absolute Q-statistic scores over all decision outputs.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Pairwise correlation score.
    """
    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
        true_assignments = multilabel_to_multiclass_assignments(true_assignments)
    return __pairwise_avg_score(decision_tensor, true_assignments, abs_q_statistic)


def pairwise_kappa_statistic(decision_tensor, true_assignments):
    """
    Calculate the average of pairwise Kappa scores over all decision outputs.
    Multilabel class assignments are transformed to equivalent multiclass class assignments.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Pairwise kappa score.
    """
    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
        true_assignments = multilabel_to_multiclass_assignments(true_assignments)
    return __pairwise_avg_score(decision_tensor, true_assignments, kappa_statistic)


def pairwise_disagreement(decision_tensor, true_assignments):
    """
    Calculate the average of pairwise disagreement scores over all decision outputs.
    Multilabel class assignments are transformed to equivalent multiclass class assignments.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Pairwise disagreement score.
    """
    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
        true_assignments = multilabel_to_multiclass_assignments(true_assignments)
    return __pairwise_avg_score(decision_tensor, true_assignments, disagreement)


def pairwise_double_fault(decision_tensor, true_assignments):
    """
    Calculate the average of pairwise double fault scores over all decision outputs.
    Multilabel class assignments are transformed to equivalent multiclass class assignments.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of crisp class assignments which are considered as true.
    :return: Pairwise double fault score.
    """
    if determine_problem(decision_tensor) == Problem.MULTI_LABEL:
        decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
        true_assignments = multilabel_to_multiclass_assignments(true_assignments)
    return __pairwise_avg_score(decision_tensor, true_assignments, double_fault)


def pairwise_euclidean_distance(decision_tensor):
    """
    Calculate the average of pairwise euclidean distance between decision matrices for the given classifiers.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multiclass decision outputs by different classifiers per sample.
    :return: Pairwise euclidean distance.
    """
    decision_tensor = np.array(decision_tensor)
    indices = np.array(np.triu_indices(decision_tensor.shape[0], k=1))
    scores = []
    for i, j in zip(indices[0], indices[1]):
        scores.append(np.mean(np.linalg.norm(decision_tensor[i] - decision_tensor[j], axis=1)))
    return np.mean(scores)
