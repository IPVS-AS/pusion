import numpy as np
from pusion.util.constants import Problem
from sklearn.metrics import *
import torch, torchmetrics
from pusion.auto.detector import determine_problem
from pusion.util.transformer import multiclass_assignments_to_labels, multilabel_to_multiclass_assignments


##################################################################################################################
###### TBC end ###################################################################################################
##################################################################################################################

# Y_predictions = curr_data['Y_predictions']
# Y_arg_max_predictions = np.zeros_like(Y_predictions)
# Y_arg_max_predictions[np.array(range(len(Y_predictions[:,1]))), np.argmax(Y_predictions, axis=1)] = 1
# Y_predictions_classes = np.argmax(Y_predictions, axis=1)
#
# Y_test = curr_data['Y_test']
# Y_test_classes = np.argmax(Y_test, axis=1)


# TODO TBC
def multi_label_brier_score_micro(y_target, y_predictions):
    '''
    Calculated the brier score for multiclass problems according to Brier 1950
    :param y_target:
    :param y_predictions:
    :return:
    '''

    y_pred_flatten = y_predictions.flatten()
    y_target_flatten = y_target.flatten()
    a = np.sum((y_pred_flatten - y_target_flatten)**2)
    a = a/y_pred_flatten.shape[0]

    c = brier_score_loss(y_target_flatten, y_pred_flatten)

    return a



# TODO TBC Brier score multi-label
def multi_label_brier_score(y_target, y_predictions):
    '''
    Calculated the brier score for multiclass problems according to Brier 1950
    :param y_target:
    :param y_predictions:
    :return:
    '''

    a = (y_predictions-y_target)**2
    a = np.sum(a, axis=1)
    a = np.sum(a)
    a = a/y_target.shape[0]

    b = np.mean(np.sum((y_predictions - y_target) ** 2, axis=1))

    return b



# TODO TBC Brier score multiclass
def multiclass_brier_score(y_target, y_predictions):
    '''
    Calculated the brier score for multiclass problems according to Brier 1950
    :param y_target:
    :param y_predictions:
    :return:
    '''

    a = (y_predictions-y_target)**2
    a = np.sum(a, axis=1)
    a = np.sum(a)
    a = a/y_target.shape[0]

    b = np.mean(np.sum((y_predictions - y_target) ** 2, axis=1))

    return b



# TODO FAR for multi-label AND multiclass classification, check the formular for correctness
def getFAR(y, y_pred, pos_normal_class) -> float:
    # False Alarm rate
    # FAR = (number of normal class samples incorrectly classified)/(number of all normal class samples) * 100

    y_normal = y[:, pos_normal_class]
    y_pred_normal = y_pred[:, pos_normal_class]

    temp_vec = y_normal * y_pred_normal
    num_all_normal_class_samples = np.sum(y_normal)
    num_normal_class_samples_incorrectly_classified = num_all_normal_class_samples - np.sum(temp_vec)

    # for testing
    # yr = yr = np.array([y_normal, y_pred_normal])
    # a = np.unique(yr, axis=1, return_counts=True)

    far = (num_normal_class_samples_incorrectly_classified / num_all_normal_class_samples)

    return far



# TODO FDR for multi-label and multiclass classification TBC , check formular for correctness
def getFDR(y, y_pred, pos_normal_class, type: str = None, counting: str = None) -> float:
    '''
    fault detection rate = (# correctly classified faulty samples) / (# all faulty samples) * 100
    In multilabel classification, the function considers the faulty subset, i. e., if the entire set
    of predicted faulty labels for a sample strictly match with the true set of faulty labels.
    :param y:
    :param y_pred:
    :param pos_normal_class:
    :param type:
    :return:
    '''

    if type == 'multiclass':
        faulty_samples_indeces = np.where(y[:, pos_normal_class] == 0)[0]
        faulty_samples = y[faulty_samples_indeces, :]
        pred_samples_at_faulty_indices = y_pred[faulty_samples_indeces, :]

        total_num_of_faulty_samples = len(faulty_samples_indeces)

        a = faulty_samples != pred_samples_at_faulty_indices
        b = np.sum(a, axis=1)
        uniques, counts = np.unique(b, return_counts=True)
        numbers_of_preds = dict(zip(uniques, counts))

        num_of_correctly_classified_faulty_samples = numbers_of_preds[0]

        fdr = (num_of_correctly_classified_faulty_samples / total_num_of_faulty_samples)

    elif type == 'multi-label':
        if counting == 'subset':
            faulty_samples_indeces = np.where(y[:, pos_normal_class] == 0)[0]
            faulty_samples = y[faulty_samples_indeces, :]
            pred_samples_at_faulty_indices = y_pred[faulty_samples_indeces, :]

            # check faulty samples
            fs1 = faulty_samples[:, 0:pos_normal_class]
            fs2 = faulty_samples[:, pos_normal_class+1:]
            fs = np.concatenate([fs1, fs2], axis=1)

            # make sure that no sample with ['normal', 0, 0, 0, ..., 0] is contained
            # --> depends on the data set if label 'normal' == [0, 0, ..., 0]
            # --> we assume that label 'normal' == [1, 0, 0, ..., 0]
            fs_sum = np.sum(fs, axis=1)
            fs_indices = np.where(fs_sum > 0)
            fs = fs[fs_indices[0], :]

            if fs.shape[0] != faulty_samples.shape[0]:
                print("Not same length!")

            #fs = faulty_samples #######
            total_num_of_faulty_samples = fs.shape[0]

            fs_pred = pred_samples_at_faulty_indices[fs_indices[0], :]
            fpreds1 = fs_pred[:, 0:pos_normal_class]
            fpreds2 = fs_pred[:, pos_normal_class + 1:]
            fpreds = np.concatenate([fpreds1, fpreds2], axis=1)

            #fpreds = pred_samples_at_faulty_indices ##########
            a = fs != fpreds
            b = np.sum(a, axis=1)
            uniques, counts = np.unique(b, return_counts=True)
            numbers_of_preds = dict(zip(uniques, counts))

            num_of_correctly_classified_faulty_samples = numbers_of_preds[0]
            fdr = (num_of_correctly_classified_faulty_samples / total_num_of_faulty_samples)

        elif counting == 'minor':
            faulty_samples_indeces = np.where(y[:, pos_normal_class] == 0)[0]
            faulty_samples = y[faulty_samples_indeces, :]
            pred_samples_at_faulty_indices = y_pred[faulty_samples_indeces, :]

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
                num_correctly_classified_faulty_labels = num_correctly_classified_faulty_labels + np.sum(temp_mult, axis=0)

            fdr = (num_correctly_classified_faulty_labels/num_faulty_labels)

        else:
            args = ('subset', 'minor')
            raise ValueError(f"Expected argument `type` to be one of {args}" f" but got {type}")
    else:
        args = ('multiclass', 'multi-label')
        raise ValueError(f"Expected argument `type` to be one of {args}" f" but got {type}")

    return fdr



# TODO TBD precision metric with average 'weighted' for multiclass
def multiclass_weighted_precision(y_true, y_pred):
    """
    TBD
    :param y_true:
    :param y_pred:
    :return:
    """
    return precision_score(Y_test_classes, Y_predictions_classes, average='weighted')



# TODO TBD precision metric with average 'weighted' for multilabel
precision_weighted = precision_score(y_true=Y_test, y_pred=Y_predictions_rounded, average='weighted')



# TODO TBD class-wise precision metric with average 'None' for multiclass
precision_class_wise = sk.metrics.precision_score(Y_test_classes, Y_predictions_classes, average=None)


# TODO TBD class-wise precision metric with average 'None' for multi-label
precision_class_wise = precision_score(y_true=Y_test, y_pred=Y_predictions_rounded, average=None)



# TODO TBD recall metric with average 'weighted' for multiclass
sk.metrics.recall_score(Y_test_classes, Y_predictions_classes, average='weighted') # Weighted recall is equal to accuracy. Cf. sk learn doc


# TODO TBD recall metric with average 'weighted' for multilabel
recall_weighted = recall_score(y_true=Y_test, y_pred=Y_predictions_rounded, average='weighted')  # Weighted recall is equal to accuracy. Cf. sk learn doc



# TODO TBD class-wise recall metric with average 'None' for multiclass
recall_class_wise = sk.metrics.recall_score(Y_test_classes, Y_predictions_classes, average=None)



# TODO TBD class-wise recall metric with average 'None' for multi-label
recall_class_wise = recall_score(y_true=Y_test, y_pred=Y_predictions_rounded, average=None)



# TODO roc auc score multiclass with average 'weighted
roc_auc_weighted_ovr = sk.metrics.roc_auc_score(y_true=Y_test_classes, y_score=Y_predictions, average='weighted', multi_class='ovr')



# TODO roc auc score multi-label with average 'weighted'
# pytorch auc roc metric
pytorch_auc_roc_weighted = torchmetrics.functional.auroc(Y_predictions_torch,
                                                         Y_target_torch,
                                                         num_classes=num_classes,
                                                         average='weighted').numpy().item()


# TODO roc auc score multi-label with average 'None'
# pytorch auc roc metric
pytorch_auc_roc_class_wise = torchmetrics.functional.auroc(Y_predictions_torch,
                                                           Y_target_torch,
                                                           num_classes=num_classes,
                                                           average=None)
pytorch_auc_roc_class_wise = [val.numpy().item() for val in pytorch_auc_roc_class_wise]


# TODO class wise average precision multiclass
pytorch_average_precision_score = torchmetrics.functional.average_precision(Y_pred_torch,
                                                                            Y_target_class_torch,
                                                                            num_classes=num_classes,
                                                                            average=None)
pytorch_average_precision_score = [val.numpy() for val in pytorch_average_precision_score]
pytorch_average_precision_score = [npa.item() for npa in pytorch_average_precision_score]


# TODO weighted average precision multiclass
pytorch_average_precision_score_weighted = torchmetrics.functional.average_precision(Y_pred_torch,
                                                                                     Y_target_class_torch,
                                                                                     num_classes=num_classes,
                                                                                     average='weighted')
pytorch_average_precision_score_weighted = pytorch_average_precision_score_weighted.numpy().item()



# TODO pytorch auc pr curve multiclass
# pytorch auc pr curve
def multiclass_auc_precisicion_recall_curve(y_true : np.ndarray, y_pred):

    Y_pred_torch = torch.from_numpy(y_pred)
    Y_target_torch = torch.from_numpy(Y_test)

    Y_pred_class_torch = torch.from_numpy(Y_predictions_classes)
    Y_target_class_torch = torch.from_numpy(Y_test_classes)

    pytorch_pr_curve_precison, pytorch_pr_curve_recall, pytorch_pr_curve_thres = torchmetrics.functional.precision_recall_curve(Y_pred_torch,
                                                                                                                                Y_target_class_torch,
                                                                                                                                num_classes=num_classes)
    pytorch_auc_pr_curve_class_wise = dict()
    for ten in range(len(pytorch_pr_curve_precison)):
        pytorch_auc_pr_curve_class_wise[ten] = torchmetrics.functional.auc(pytorch_pr_curve_recall[ten],
                                                                         pytorch_pr_curve_precison[ten]).numpy().item()
    agg_test_pytorch_auc_pr_curve_class_wise.append(pytorch_auc_pr_curve_class_wise)



# TODO pytorch auc roc metric multiclass
pytorch_auc_roc_weighted = torchmetrics.functional.auroc(Y_pred_torch,
                                                         Y_target_class_torch,
                                                         num_classes=num_classes,
                                                         average='weighted').numpy().item()

pytorch_auc_roc_class_wise = torchmetrics.functional.auroc(Y_pred_torch,
                                                           Y_target_class_torch,
                                                           num_classes=num_classes,
                                                           average=None)
pytorch_auc_roc_class_wise = [val.numpy().item() for val in pytorch_auc_roc_class_wise]



#TODO label ranking average precision score for multi-label
test_lraps = label_ranking_average_precision_score(y_true=Y_test, y_score=Y_predictions)


#TODO label ranking loss multi-label
test_multilabel_ranking_loss = label_ranking_loss(y_true=Y_test, y_score=Y_predictions)


# TODO Normalized Discounted Cumulative Gain multi-label
test_ndcg = ndcg_score(y_true=Y_test, y_score=Y_predictions)


# TODO  top-1 accuracy for multiclass
top_1_acc = sk.metrics.top_k_accuracy_score(Y_test_classes, Y_predictions, k=1, normalize=True, sample_weight=None, labels=None)


# TODO top-3 accuracy for multiclass
top_3_acc = sk.metrics.top_k_accuracy_score(Y_test_classes, Y_predictions, k=3, normalize=True, sample_weight=None, labels=None)


# TODO  top-5 accuracy for multiclass
def multiclass_top_5_accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    if len(y_true.shape) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])

    top_5_acc = top_k_accuracy_score(y_true, y_pred, k=5, normalize=True, sample_weight=None, labels=None)
    return top_5_acc


# TODO multiclass log loss
log_loss = sk.metrics.log_loss(Y_test_classes, Y_predictions)


# TODO multi-label log loss
log_loss = sk.metrics.log_loss(Y_test_classes, Y_predictions)


##################################################################################################################
###### TBC end ###################################################################################################
##################################################################################################################

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


# TODO implement False Alarm Rate metric as described in Tidriri et al. 2018

# TODO implement Fault Detection Rate metric as described in Tidriri et al. 2018
