import multiprocessing as mp

from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split

from pusion.util.processes import p_fit
from pusion.util.transformer import *


def run_multiclass_ensemble(classifiers, x_train, y_train, x_valid, y_valid, x_test, y_test, continuous_out=False,
                            parallelize=True):
    """
    TODO comments.
    Generate random multiclass, crisp and redundant classification outputs (assignments) for the given ensemble of
    classifiers.

    :param classifiers: Classifiers used to generate classification outputs.
            These need to implement `fit` and `predict` methods according to classifiers provided by `sklearn`.
    :param parallelize: If `True`, all classifiers are trained in parallel. Otherwise they are trained in sequence.
    :param continuous_out: If `True`, class assignments in `y_ensemble_valid` and `y_ensemble_test` are given as
            probabilities. Default value is `False`.
    :return: `tuple` of:
            - `y_ensemble_valid`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a validation dataset.
            - `y_valid`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the validation.
            - `y_ensemble_test`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a test dataset.
            - `y_test`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the test.
    """

    if not (y_train.shape[1] == y_valid.shape[1] == y_test.shape[1]):
        raise TypeError("Different number of classes across train, valid and test.")

    n_classes = y_train.shape[1]

    # Convert to label vector to train sklearn classifiers (requires pusion commit 39a687 from master branch).
    y_train = class_assignment_matrix_to_label_vector(y_train)

    # Classifier training
    if parallelize:
        # Parallelized classifier training
        # Create a thread-safe queue for classifiers
        queue = mp.Manager().Queue()
        processes = []
        for i, clf in enumerate(classifiers):
            processes.append(mp.Process(target=p_fit, args=(i, clf, x_train, y_train, queue)))
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        while not queue.empty():
            i, clf = queue.get()
            classifiers[i] = clf
    else:
        # Sequential classifier training
        for i, clf in enumerate(classifiers):
            print("Train classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            clf.fit(x_train, y_train)

    # Classifier validation to generate combiner training data
    y_ensemble_valid = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_ensemble_valid.append(clf.predict_proba(x_valid))
    else:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_ensemble_valid.append(clf.predict(x_valid))

    # Classifier test
    y_ensemble_test = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_ensemble_test.append(clf.predict_proba(x_test))
    else:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_ensemble_test.append(clf.predict(x_test))

    # Transform to matrix representation
    if not continuous_out:
        y_ensemble_valid = transform_label_tensor_to_class_assignment_tensor(y_ensemble_valid, n_classes)
        y_ensemble_test = transform_label_tensor_to_class_assignment_tensor(y_ensemble_test, n_classes)

    # Transform to numpy tensors if possible
    y_ensemble_valid = tensorize(y_ensemble_valid)
    y_ensemble_test = tensorize(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test
