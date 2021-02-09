import numpy as np

from clunpy.constants import Metric, Native
from clunpy.transformer import confusion_matrix_to_accuracy, multiclass_predictions_to_decisions


def convert_native_to_clunpy_data(dataset):
    # accumulate with lists to make use of python's over-allocation.
    # (see https://github.com/numpy/numpy/issues/17090)
    classifier_evidence_vector = []
    classifier_prediction_tensor = []
    train_predictions = []
    true_class_assignments = None

    for d in dataset:
        # Extract evidence
        if Metric.CONFUSION_MATRIX in d:
            classifier_evidence_vector.append(confusion_matrix_to_accuracy(np.array(d[Metric.CONFUSION_MATRIX])))
        elif Metric.ACCURACY in d:
            classifier_evidence_vector.append(d[Metric.ACCURACY])

        # Extract classifier outputs according to the type
        if Native.PREDICTIONS in d:
            classifier_prediction_tensor.append(d[Native.PREDICTIONS])
        else:
            raise TypeError("Could not find", Native.PREDICTIONS, "in the given data set.")

        # Extract train predictions of each classifier
        if Native.TRAIN_PREDICTIONS in d:
            train_predictions.append(d[Native.TRAIN_LABELS])

        # Extract true class label assignments representing training labels
        if Native.TRAIN_LABELS in d and true_class_assignments is None:
            true_class_assignments = d[Native.TRAIN_LABELS]

    clunpy_data = {
        "decision_outputs_tensor": np.array(classifier_prediction_tensor),
        "evidence": np.array(classifier_evidence_vector),
        "train_predictions": np.array(train_predictions),
        "true_assignments": np.array(true_class_assignments)
    }

    return clunpy_data
