import numpy as np
#import torch
import torchmetrics
from sklearn.metrics import *

from pusion.auto.detector import determine_problem
from pusion.util.constants import Problem
from pusion.util.transformer import multiclass_assignments_to_labels, multilabel_to_multiclass_assignments


def multi_label_brier_score_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the brier score for multi-label problems according to Brier 1950
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The micro brier score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    y_pred_flatten = y_pred.flatten()
    y_target_flatten = y_true.flatten()
    result = np.sum((y_pred_flatten - y_target_flatten) ** 2) / y_pred_flatten.shape[0]
    result = result / y_pred_flatten.shape[0]

    return result


def multi_label_brier_score(y_true: np.ndarray, y_pred: np.ndarray):  # -> float: #TODO
    """
    Calculate the brier score for multiclass problems according to Brier 1950
    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The brier score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    result = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    return result


def multiclass_brier_score(y_true: np.ndarray, y_pred: np.ndarray):  # -> float:
    """
    Calculate the brier score for multi-label problems according to Brier 1950
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The brier score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    result = np.mean(np.sum((y_pred - y_true) ** 2, axis=1))

    return result


def far(y_true: np.ndarray, y_pred: np.ndarray, pos_normal_class: int = 0) -> float:
    """
    Calculate the false alarm rate for multiclass and multi-label problems.
    FAR = (number of normal class samples incorrectly classified)/(number of all normal class samples) * 100
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :param pos_normal_class: the position of the 'normal class' in :param y_true and :param y_pred. Default is `0`
    :return: The false alarm rate.
    """
    # False Alarm rate
    # FAR = (number of normal class samples incorrectly classified)/(number of all normal class samples) * 100

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    y_true_normal = y_true[:, pos_normal_class]
    y_pred_normal = y_pred[:, pos_normal_class]

    temp_vec = y_true_normal * y_pred_normal
    num_all_normal_class_samples = np.sum(y_true_normal)
    num_normal_class_samples_incorrectly_classified = num_all_normal_class_samples - np.sum(temp_vec)

    # for testing
    # yr = yr = np.array([y_true_normal, y_pred_normal])
    # a = np.unique(yr, axis=1, return_counts=True)
    far = (num_normal_class_samples_incorrectly_classified / num_all_normal_class_samples)
    return far


def multiclass_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    fault detection rate = (# correctly classified faulty samples) / (# all faulty samples) * 100
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The fault detection rate.
    """
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    pos_normal_class = 0
    faulty_samples_indices = np.where(y_true[:, pos_normal_class] == 0)[0]
    faulty_samples = y_true[faulty_samples_indices, :]
    pred_samples_at_faulty_indices = y_pred[faulty_samples_indices, :]

    total_num_of_faulty_samples = len(faulty_samples_indices)

    a = faulty_samples != pred_samples_at_faulty_indices
    b = np.sum(a, axis=1)
    uniques, counts = np.unique(b, return_counts=True)
    numbers_of_preds = dict(zip(uniques, counts))

    num_of_correctly_classified_faulty_samples = numbers_of_preds[0]

    fdr = (num_of_correctly_classified_faulty_samples / total_num_of_faulty_samples)
    return fdr


def multilabel_subset_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    fault detection rate = (# correctly classified faulty samples) / (# all faulty samples) * 100
    In multilabel classification, the function considers the faulty subset, i. e., if the entire set
    of predicted faulty labels for a sample strictly match with the true set of faulty labels.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The fault detection rate.
    """
    # TODO(old): counting: 'subset' or 'minor'. The way how the faulty subsets should be counted. TODO correct?

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    pos_normal_class = 0
    faulty_samples_indices = np.where(y_true[:, pos_normal_class] == 0)[0]
    faulty_samples = y_true[faulty_samples_indices, :]
    pred_samples_at_faulty_indices = y_pred[faulty_samples_indices, :]

    # check faulty samples
    fs1 = faulty_samples[:, 0:pos_normal_class]
    fs2 = faulty_samples[:, pos_normal_class + 1:]
    fs = np.concatenate([fs1, fs2], axis=1)

    # make sure that no sample with ['normal', 0, 0, 0, ..., 0] is contained
    # --> depends on the data set if label 'normal' == [0, 0, ..., 0]
    # --> we assume that label 'normal' == [1, 0, 0, ..., 0]
    fs_sum = np.sum(fs, axis=1)
    fs_indices = np.where(fs_sum > 0)
    fs = fs[fs_indices[0], :]

    if fs.shape[0] != faulty_samples.shape[0]:
        print("Not same length!")

    # fs = faulty_samples #######
    total_num_of_faulty_samples = fs.shape[0]

    fs_pred = pred_samples_at_faulty_indices[fs_indices[0], :]
    fpreds1 = fs_pred[:, 0:pos_normal_class]
    fpreds2 = fs_pred[:, pos_normal_class + 1:]
    fpreds = np.concatenate([fpreds1, fpreds2], axis=1)

    # fpreds = pred_samples_at_faulty_indices ##########
    a = fs != fpreds
    b = np.sum(a, axis=1)
    uniques, counts = np.unique(b, return_counts=True)
    numbers_of_preds = dict(zip(uniques, counts))

    num_of_correctly_classified_faulty_samples = numbers_of_preds[0]
    fdr = (num_of_correctly_classified_faulty_samples / total_num_of_faulty_samples)
    return fdr


def multilabel_minor_fdr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    fault detection rate = (# correctly classified faulty samples) / (# all faulty samples) * 100
    In multilabel classification, the function considers the faulty subset, i. e., if the entire set
    of predicted faulty labels for a sample strictly match with the true set of faulty labels.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The fault detection rate.
    """
    # TODO(old): counting: 'subset' or 'minor'. The way how the faulty subsets should be counted. TODO correct?

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    pos_normal_class = 0
    faulty_samples_indices = np.where(y_true[:, pos_normal_class] == 0)[0]
    faulty_samples = y_true[faulty_samples_indices, :]
    pred_samples_at_faulty_indices = y_pred[faulty_samples_indices, :]

    # check faulty samples
    fs1 = faulty_samples[:, 0:pos_normal_class]
    fs2 = faulty_samples[:, pos_normal_class + 1:]
    fs = np.concatenate([fs1, fs2], axis=1)

    fs_preds1 = pred_samples_at_faulty_indices[:, 0:pos_normal_class]
    fs_preds2 = pred_samples_at_faulty_indices[:, pos_normal_class + 1:]
    fs_preds = np.concatenate([fs_preds1, fs_preds2], axis=1)

    num_faulty_labels = 0
    num_correctly_classified_faulty_labels = 0
    for col in range(fs.shape[1]):
        fs_col = fs[:, col]
        fs_preds_col = fs_preds[:, col]

        num_faulty_labels = num_faulty_labels + np.sum(fs_col, axis=0)
        temp_mult = fs_col * fs_preds_col
        num_correctly_classified_faulty_labels = num_correctly_classified_faulty_labels + np.sum(temp_mult,
                                                                                                 axis=0)

    fdr = (num_correctly_classified_faulty_labels / num_faulty_labels)
    return fdr


def multiclass_weighted_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the precision for a multiclass problem with a `weighted` average.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 1 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 1D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return precision_score(y_true, y_pred, average='weighted')


def multi_label_weighted_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the precision for a multi-label problem with a `weighted` average.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return precision_score(y_true, y_pred, average='weighted')


def multiclass_class_wise_precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the precision for a multiclass problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    y_true = multiclass_assignments_to_labels(y_true)
    y_pred = multiclass_assignments_to_labels(y_pred)

    return precision_score(y_true, y_pred, average=None)


def multi_label_class_wise_precision(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the precision for a multi-label problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return precision_score(y_true, y_pred, average=None)


def multiclass_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the recall for a multiclass problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)`. Predicted labels or class assignments.
    :return: The recall score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 1 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 1D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return recall_score(y_true, y_pred,
                        average='weighted')  # Weighted recall is equal to accuracy. Cf. sk learn doc


def multi_label_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the recall for a multi-label problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The recall score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return recall_score(y_true, y_pred, average='weighted')  # Weighted recall is equal to accuracy. Cf. sk learn doc


def multiclass_class_wise_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the recall for a multiclass problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: Sequence of recall scores (for each class).
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    y_true = multiclass_assignments_to_labels(y_true)
    y_pred = multiclass_assignments_to_labels(y_pred)

    return recall_score(y_true, y_pred, average=None)


def multi_label_class_wise_recall(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate the recall for a multi-label problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: Sequence of recall scores (for each class).
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return recall_score(y_true, y_pred, average=None)

# TODO float
def multiclass_weighted_scikit_auc_roc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the scikit auc roc score for a multi-label problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The auc roc score.
    """

    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true and y_pred needs to be 2D with same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    # y_true = multiclass_assignments_to_labels(y_true)

    return roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

# TODO float
def multi_label_weighted_pytorch_auc_roc_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the pytorch auc roc score for a multi-label problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The auc roc score.
    """
    # pytorch auc roc metric
    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, y_pred needs to have dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    y_pred = y_pred.astype(np.float)
    num_classes = y_true.shape[1]
    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    return torchmetrics.functional.auroc(y_pred_torch, y_true_torch, num_classes=num_classes,
                                         average='weighted').numpy().item()

# TODO float
def multi_label_pytorch_auc_roc_score(y_true: np.ndarray, y_pred: np.ndarray): # -> list[float]:
    """
    Compute the pytorch auc roc score for a multi-label problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The auc roc score.
    """
    # pytorch auc roc metric
    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, y_pred needs to have dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )
    y_pred = y_pred.astype(np.float)
    num_classes = y_true.shape[1]
    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    pytorch_auc_roc_class_wise = torchmetrics.functional.auroc(y_pred_torch, y_true_torch, num_classes=num_classes,
                                                               average=None)
    return [val.numpy().item() for val in pytorch_auc_roc_class_wise]

# TODO float
def multiclass_class_wise_avg_precision(y_true: np.ndarray, y_pred: np.ndarray): # -> list[float]:
    """
    Compute the class wise precision for a multiclass problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true and y_pred need to be 2D with the same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    num_classes = y_true.shape[1]
    y_true = multiclass_assignments_to_labels(y_true)

    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    pytorch_average_precision_score = torchmetrics.functional.average_precision(y_pred_torch,
                                                                                y_true_torch,
                                                                                num_classes=num_classes,
                                                                                average=None)
    pytorch_average_precision_score = [val.numpy() for val in pytorch_average_precision_score]
    return [npa.item() for npa in pytorch_average_precision_score]

# TODO float
def multiclass_weighted_avg_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the precision for a multiclass problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
        raise ValueError(
            "y_true needs to be 1D and y_pred needs to be 2D with same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    num_classes = y_true.shape[1]
    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    pytorch_average_precision_score_weighted = torchmetrics.functional.average_precision(y_pred_torch,
                                                                                         y_true_torch,
                                                                                         num_classes=num_classes,
                                                                                         average='weighted')
    return pytorch_average_precision_score_weighted.numpy().item()


# TODO float
def multiclass_auc_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray):  # -> list[dict[int, float]]:
    """
    Compute the class wise auc precision recall curve for a multiclass problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The aggregated auc precision recall curve class wise.
    """

    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true and y_pred need to be 2D with the same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    num_classes = y_true.shape[1]
    y_true = y_true.astype(np.float)
    y_true = multiclass_assignments_to_labels(y_true)
    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    pytorch_pr_curve_precision, pytorch_pr_curve_recall, pytorch_pr_curve_thres = torchmetrics.functional.precision_recall_curve(
        y_pred_torch,
        y_true_torch,
        num_classes=num_classes)

    agg_test_pytorch_auc_pr_curve_class_wise = []
    np_agg_test_pytorch_auc_pr_curve_class_wise = np.full(len(pytorch_pr_curve_precision), np.nan)
    pytorch_auc_pr_curve_class_wise = dict()
    for ten in range(len(pytorch_pr_curve_precision)):
        val = torchmetrics.functional.auc(pytorch_pr_curve_recall[ten], pytorch_pr_curve_precision[ten]).numpy().item()
        pytorch_auc_pr_curve_class_wise[ten] = val
        np_agg_test_pytorch_auc_pr_curve_class_wise[ten] = val
    agg_test_pytorch_auc_pr_curve_class_wise.append(pytorch_auc_pr_curve_class_wise)

    return np_agg_test_pytorch_auc_pr_curve_class_wise

# TODO float
def multiclass_weighted_pytorch_auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the pytorch auc roc for a multiclass problem with average `weighted`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The auc roc score.
    """

    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true and y_pred needs to be 2D with same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )

    num_classes = y_true.shape[1]
    y_true = y_true.astype(np.float)
    y_true = multiclass_assignments_to_labels(y_true)
    y_pred = y_pred.astype(np.float)

    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    return torchmetrics.functional.auroc(y_pred_torch,
                                         y_true_torch,
                                         num_classes=num_classes,
                                         average='weighted').numpy().item()

# TODO float
def multiclass_pytorch_auc_roc(y_true: np.ndarray, y_pred: np.ndarray): # -> list[float]:
    """
    Compute the pytorch auc roc for a multiclass problem with average `None`.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The auc roc score.
    """

    # TODO: quick fix done here.
    # if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0] and np.issubdtype(y_pred.dtype, np.floating)):
    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true and y_pred needs to be 2D with the same number of samples and dtype float, "
            "got {} and {} with dtype {} instead.".format(y_true.shape, y_pred.shape, y_pred.dtype)
        )
    num_classes = y_true.shape[1]
    y_true = y_true.astype(np.float)
    y_true = multiclass_assignments_to_labels(y_true)
    y_pred = y_pred.astype(np.float)

    y_pred_torch = torch.from_numpy(y_pred)
    y_true_torch = torch.from_numpy(y_true)
    pytorch_auc_roc_class_wise = torchmetrics.functional.auroc(y_pred_torch,
                                                               y_true_torch,
                                                               num_classes=num_classes,
                                                               average=None)
    return [val.numpy().item() for val in pytorch_auc_roc_class_wise]


def multi_label_ranking_avg_precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the label ranking based average precision score for a multi-label problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return label_ranking_average_precision_score(y_true=y_true, y_score=y_pred)


def multi_label_ranking_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the label ranking loss for a multi-label problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The precision score.
    :return: The loss.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return label_ranking_loss(y_true=y_true, y_score=y_pred)


def multi_label_normalized_discounted_cumulative_gain(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the normalized discounted cumulative gain for a multi-label problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The gain.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return ndcg_score(y_true=y_true, y_score=y_pred)


def multiclass_top_1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the top-1 accuracy for a multiclass problem.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The accuracy score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true needs to be 1D and y_pred needs to be 2D with same number of samples, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return top_k_accuracy_score(y_true, y_pred, k=1, normalize=True, sample_weight=None,
                                labels=None)


def multiclass_top_3_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the top-3 accuracy for a multiclass problem.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The accuracy score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true needs to be 1D and y_pred needs to be 2D with same number of samples, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return top_k_accuracy_score(y_true, y_pred, k=3, normalize=True, sample_weight=None,
                                labels=None)


def multiclass_top_5_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the top-5 accuracy for a multiclass problem.
    :param y_true: `numpy.array` of shape `(n_samples,)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape (n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The accuracy score.
    """

    if not (y_true.ndim == 1 and y_pred.ndim == 2 and y_true.shape[0] == y_pred.shape[0]):
        raise ValueError(
            "y_true needs to be 1D and y_pred needs to be 2D with same number of samples, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    top_5_acc = top_k_accuracy_score(y_true, y_pred, k=5, normalize=True, sample_weight=None, labels=None)
    return top_5_acc


def multiclass_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The logarithmic loss for a multiclass problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The loss.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return log_loss(y_true, y_pred)


def multi_label_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The logarithmic loss for a multi-label problem.
    :param y_true: `numpy.array` of shape `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The loss.
    """

    if not (y_true.ndim == 2 and y_pred.ndim == 2 and y_true.shape == y_pred.shape):
        raise ValueError(
            "y_true and y_pred need to be 2D and have the same shape, "
            "got {} and {} instead.".format(y_true.shape, y_pred.shape)
        )

    return log_loss(y_true, y_pred)




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


def weighted_f1(y_true, y_pred):
    """
    Calculate the macro F1-score, i.e. 2 * (Precision * Recall) / (Precision + Recall), weighted by the class support.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: The weighted macro F1-score.
    """
    return f1_score(y_true, y_pred, average='weighted')


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


def weighted_jaccard(y_true, y_pred):
    """
    Calculate the Jaccard-score for each label, and find their average, weighted by support, i. e., the number of true instances of each label instance.

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: The macro Jaccard-score.
    """
    return jaccard_score(y_true, y_pred, average="weighted")


def accuracy(y_true, y_pred):
    """
    Calculate the accuracy, i.e. (TP + TN) / (TP + FP + FN + TN).

    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or
            class assignments.
    :return: Accuracy.
    """
    return accuracy_score(y_true, y_pred)


def error_rate(y_true, y_pred):
    """
    Calculate the error rate, i. e. error_rate = 1-accuracy
    :param y_true: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. True labels or class assignments.
    :param y_pred: `numpy.array` of shape `(n_samples,)` or `(n_samples, n_classes)`. Predicted labels or class assignments.
    :return: Error Rate of typ `float`
    """
    return 1-accuracy_score(y_true, y_pred)


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
    return a / n_samples, b / n_samples, c / n_samples, d / n_samples


def __pairwise_avg_score(decision_tensor, true_assignments, score_func, **kwargs):
    """
    A helper function for calculating pairwise average score statistics.
    """
    decision_tensor = np.array(decision_tensor)
    indices = np.array(np.triu_indices(decision_tensor.shape[0], k=1))
    scores = []
    for i, j in zip(indices[0], indices[1]):
        scores.append(score_func(decision_tensor[i], decision_tensor[j], true_assignments))

    if 'return_type' in kwargs:
        if kwargs['return_type'] == 'list':
            return indices, scores

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
    return (2 * (a * d - b * c)) / ((a + b) * (b + d) + (a + c) * (c + d))


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


def pairwise_correlation(decision_tensor, true_assignments, **kwargs):
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
    return __pairwise_avg_score(decision_tensor, true_assignments, abs_correlation, **kwargs)


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


def pairwise_kappa_statistic(decision_tensor, true_assignments, **kwargs):
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
    return __pairwise_avg_score(decision_tensor, true_assignments, kappa_statistic, **kwargs)


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


def pairwise_double_fault(decision_tensor, true_assignments, **kwargs):
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
    return __pairwise_avg_score(decision_tensor, true_assignments, double_fault, **kwargs)


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

