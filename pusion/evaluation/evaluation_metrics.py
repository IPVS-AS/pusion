import numpy as np
from sklearn.metrics import *

from pusion.util.transformer import multiclass_assignments_to_labels, multilabel_to_multiclass_assignments


def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, average='micro')


def recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average='micro')


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')


def jaccard(y_true, y_pred):
    return jaccard_score(y_true, y_pred, average='micro')


def mean_multilabel_accuracy(y_true, y_pred):
    np.sum(multilabel_confusion_matrix(y_true, y_pred), axis=0) / len(y_pred)


def hamming(y_true, y_pred):
    return hamming_loss(y_true, y_pred)


def log(y_true, y_pred):
    return log_loss(y_true, y_pred)


def cohens_kappa(y1, y2, labels):
    cm = confusion_matrix(y1, y2, labels=labels)
    a = np.sum(np.diagonal(cm)) / np.sum(cm)
    e = 0
    for i in range(len(cm)):
        e += np.sum(cm[i, :]) * np.sum(cm[:, i]) / np.sum(cm) ** 2
    if e == 1:
        return 1.0  # case when y1 and y2 are equivalent in their annotation
    return (a - e) / (1 - e)


def pairwise_cohens_kappa_multiclass(decision_tensor):
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
    mc_decision_tensor = multilabel_to_multiclass_assignments(decision_tensor)
    return pairwise_cohens_kappa_multiclass(mc_decision_tensor)


def pairwise_micro_score(decision_tensor, score_func):
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
    def q_stat_func(a, b, c, d): return (a * d - b * c) / (a * d + b * c)
    return pairwise_micro_score(decision_tensor, q_stat_func)


def correlation(decision_tensor):
    def correlation_func(a, b, c, d): return (a * d - b * c) / np.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    return pairwise_micro_score(decision_tensor, correlation_func)
