import multiprocessing as mp

from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split

from pusion.util.processes import p_fit
from pusion.util.transformer import *


def generate_multiclass_ensemble_classification_outputs(classifiers, n_classes, n_samples, continuous_out=False,
                                                        parallelize=True):
    """
    Generate random multiclass, crisp and redundant classification outputs (assignments) for the given ensemble of
    classifiers.

    :param classifiers: Classifiers used to generate classification outputs.
            These need to implement `fit` and `predict` methods according to classifiers provided by `sklearn`.
    :param n_classes: `integer`. Number of classes, predictions are made for.
    :param n_samples: `integer`. Number of samples.
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
    x, y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_redundant=0,
                               n_informative=n_classes,
                               n_clusters_per_class=1)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

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
    y_valid = transform_label_vector_to_class_assignment_matrix(y_valid, n_classes)
    y_test = transform_label_vector_to_class_assignment_matrix(y_test, n_classes)

    # Transform to numpy tensors if possible
    y_ensemble_valid = tensorize(y_ensemble_valid)
    y_ensemble_test = tensorize(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def generate_multiclass_cr_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage=None,
                                                           continuous_out=False, parallelize=True):
    """
    Generate random multiclass, crisp and complementary-redundant classification outputs (assignments) for the given
    ensemble of classifiers.

    :param classifiers: Classifiers used to generate classification outputs.
            These need to implement `fit` and `predict` methods according to classifiers provided by `sklearn`.
    :param n_classes: `integer`. Number of classes, predictions are made for.
    :param n_samples: `integer`. Number of samples.
    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
            If unset, redundant classification outputs are retrieved.
    :param continuous_out: If `True`, class assignments in `y_ensemble_valid` and `y_ensemble_test` are given as
            probabilities. Default value is `False`.
    :param parallelize: If `True`, all classifiers are trained in parallel. Otherwise they are trained in sequence.
    :return: `tuple` of:
            - `y_ensemble_valid`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a validation dataset.
            - `y_valid`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the validation.
            - `y_ensemble_test`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a test dataset.
            - `y_test`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the test.
    """
    x, y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_redundant=0,
                               n_informative=n_classes,
                               n_clusters_per_class=1)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

    # Transform to class assignments
    y_train = transform_label_vector_to_class_assignment_matrix(y_train, n_classes)
    y_valid = transform_label_vector_to_class_assignment_matrix(y_valid, n_classes)
    y_test = transform_label_vector_to_class_assignment_matrix(y_test, n_classes)

    # Classifier training
    if parallelize:
        # Parallelized classifier training
        # Create a thread-safe queue for classifiers
        queue = mp.Manager().Queue()
        processes = []
        for i, clf in enumerate(classifiers):
            y_train_ = y_train[:, coverage[i]]
            y_train_ = intercept_normal_class(y_train_, override=True)
            y_train_ = multiclass_assignments_to_labels(y_train_)
            processes.append(mp.Process(target=p_fit, args=(i, clf, x_train, y_train_, queue)))
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
            y_train_ = y_train[:, coverage[i]]
            y_train_ = intercept_normal_class(y_train_, override=True)
            y_train_ = multiclass_assignments_to_labels(y_train_)
            clf.fit(x_train, y_train_)

    # Classifier validation to generate combiner training data
    y_ensemble_valid = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict_proba(x_valid)
            y_ensemble_valid.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_valid)
            y_pred = transform_label_vector_to_class_assignment_matrix(y_pred, len(coverage[i]))
            y_ensemble_valid.append(y_pred)

    # Classifier test
    y_ensemble_test = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict_proba(x_test)
            y_ensemble_test.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_test)
            y_pred = transform_label_vector_to_class_assignment_matrix(y_pred, len(coverage[i]))
            y_ensemble_test.append(y_pred)

    # Transform to numpy tensors if possible
    y_ensemble_valid = tensorize(y_ensemble_valid)
    y_ensemble_test = tensorize(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def generate_multilabel_ensemble_classification_outputs(classifiers, n_classes, n_samples,  continuous_out=False,
                                                        parallelize=True):
    """
    Generate random multilabel crisp classification outputs (assignments) for the given ensemble of classifiers with
    the normal class included at index `0`.

    :param classifiers: Classifiers used to generate classification outputs.
            These need to implement `fit` and `predict` methods according to classifiers provided by `sklearn`.
    :param n_classes: `integer`. Number of classes, predictions are made for with the normal class included.
    :param n_samples: `integer`. Number of samples.
    :param continuous_out: If `True`, class assignments in `y_ensemble_valid` and `y_ensemble_test` are given as
            probabilities. Default value is `False`.
    :param parallelize: If `True`, all classifiers are trained in parallel. Otherwise they are trained in sequence.
    :return: `tuple` of:
            - `y_ensemble_valid`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a validation dataset.
            - `y_valid`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the validation.
            - `y_ensemble_test`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a test dataset.
            - `y_test`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the test.
    """
    x, y = make_multilabel_classification(n_samples=n_samples,
                                          n_classes=n_classes-1,  # -1, due to normal class
                                          n_labels=2,
                                          allow_unlabeled=True)

    # Adapt y to address the normal class representation in case of an unlabeled output (prepend normal class)
    y = intercept_normal_class(y, override=False)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

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
            y_pred = clf.predict_proba(x_valid)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_valid.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_valid)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_valid.append(y_pred)

    # Classifier test
    y_ensemble_test = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict_proba(x_test)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_test.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_test)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_test.append(y_pred)

    # Transform to numpy tensors if possible
    y_ensemble_valid = tensorize(y_ensemble_valid)
    y_ensemble_test = tensorize(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def generate_multilabel_cr_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage=None,
                                                           continuous_out=False, parallelize=True):
    """
    Generate random multilabel, crisp and complementary-redundant classification outputs (assignments) for the given
    ensemble of classifiers with the normal class included at index `0`.

    :param classifiers: Classifiers used to generate classification outputs.
            These need to implement `fit` and `predict` methods according to classifiers provided by `sklearn`.
    :param n_classes: `integer`. Number of classes, predictions are made for with the normal class included.
    :param n_samples: `integer`. Number of samples.
    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
            If unset, redundant classification outputs are retrieved.
    :param continuous_out: If `True`, class assignments in `y_ensemble_valid` and `y_ensemble_test` are given as
            probabilities. Default value is `False`.
    :param parallelize: If `True`, all classifiers are trained in parallel. Otherwise they are trained in sequence.
    :return: `tuple` of:
            - `y_ensemble_valid`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a validation dataset.
            - `y_valid`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the validation.
            - `y_ensemble_test`: `numpy.array` of shape `(n_samples, n_classes)`. Ensemble decision output matrix for
            as a test dataset.
            - `y_test`: `numpy.array` of shape `(n_samples, n_classes)`. True class assignments for the test.
    """
    x, y = make_multilabel_classification(n_samples=n_samples,
                                          n_classes=n_classes-1,  # -1, due to normal class
                                          n_labels=2,
                                          allow_unlabeled=True)

    # Adapt y to address the normal class representation in case of an unlabeled output (prepend normal class)
    y = intercept_normal_class(y, override=False)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

    # Classifier training
    if parallelize:
        # Parallelized classifier training
        # Create a thread-safe queue for classifiers
        queue = mp.Manager().Queue()
        processes = []
        for i, clf in enumerate(classifiers):
            y_train_ = y_train[:, coverage[i]]
            y_train_ = intercept_normal_class(y_train_, override=True)
            processes.append(mp.Process(target=p_fit, args=(i, clf, x_train, y_train_, queue)))
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
            y_train_ = y_train[:, coverage[i]]
            y_train_ = intercept_normal_class(y_train_, override=True)
            clf.fit(x_train, y_train_)

    # Classifier validation to generate combiner training data
    y_ensemble_valid = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict_proba(x_valid)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_valid.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_valid)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_valid.append(y_pred)

    # Classifier test
    y_ensemble_test = []
    if continuous_out:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict_proba(x_test)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_test.append(y_pred)
    else:
        for i, clf in enumerate(classifiers):
            print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
            y_pred = clf.predict(x_test)
            # y_pred = intercept_normal_class(y_pred, override=True)
            y_ensemble_test.append(y_pred)

    # Transform to numpy tensors if possible
    y_ensemble_valid = tensorize(y_ensemble_valid)
    y_ensemble_test = tensorize(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def generate_multiclass_confusion_matrices(decision_tensor, true_assignments):
    """
    Generate multiclass confusion matrices out of the given decision tensor and true assignments.
    Continuous outputs are converted to multiclass assignments using the MAX rule.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for calculating confusion matrices.
    :return: `numpy.array` of shape `(n_classifiers, n_samples, n_samples)`. Confusion matrices per classifier.
    """
    true_assignment_labels = np.argmax(true_assignments, axis=1)
    confusion_matrices = np.zeros((np.shape(decision_tensor)[0],
                                   np.shape(true_assignments)[1],
                                   np.shape(true_assignments)[1]), dtype=int)
    for i in range(len(decision_tensor)):
        decision_tensor_labels = np.argmax(decision_tensor[i], axis=1)
        confusion_matrices[i] = confusion_matrix(y_true=true_assignment_labels,
                                                 y_pred=decision_tensor_labels,
                                                 labels=np.arange(np.shape(true_assignments)[1]))
    return confusion_matrices


def generate_multilabel_cr_confusion_matrices(decision_outputs, true_assignments, coverage):
    """
    Generate multilabel confusion matrices for complementary-redundant multilabel classification outputs.

    :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for calculating confusion matrices.
    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
    :return: List of multilabel confusion matrices.
    """
    cr_confusion_matrices = []
    for i, do in enumerate(decision_outputs):
        ta = intercept_normal_class(true_assignments[:, coverage[i]], override=True)
        cms = multilabel_confusion_matrix(y_true=ta, y_pred=do, labels=np.arange(len(coverage[i])))
        cr_confusion_matrices.append(cms)
    return cr_confusion_matrices


def generate_classification_coverage(n_classifiers, n_classes, overlap, normal_class=True):
    """
    Generate random complementary redundant class indices for each classifier `0..(n_classifiers-1)`.
    The coverage is drawn from normal distribution for all classifiers.
    However, it is guaranteed that each classifier covers at least one class regardless of the distribution.

    :param n_classifiers: Number of classifiers representing the classifier `0..(n_classifiers-1)`.
    :param n_classes: Number of classes representing the class label `0..(n_classes-1)`.
    :param overlap: Indicator between `0` and `1` for overall classifier overlapping in terms of classes.
            If `0`, only complementary class indices are obtained.
            If `1`, the overlapping is fully redundant.
    :param normal_class: If `True`, a class for the normal state is included for all classifiers as class index `0`.
    :return: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
    """
    if normal_class:
        n_classes = n_classes - 1
    coverage_matrix = np.zeros((n_classifiers, n_classes), dtype=int)
    n_selected_classifier = int(np.interp(overlap, [0, 1], [np.ceil(n_classifiers/n_classes), n_classifiers]))
    while np.any(coverage_matrix.sum(axis=1) < 1):
        coverage_matrix = np.zeros((n_classifiers, n_classes), dtype=int)
        for i in range(n_classes):
            selected_classifier = np.random.choice(np.arange(n_classifiers), n_selected_classifier, replace=False)
            coverage_matrix[selected_classifier, i] = 1

    class_index_list = []
    if normal_class:
        for i in range(n_classifiers):
            class_index_list.append(np.array([0, *np.where(coverage_matrix[i])[0] + 1]))
    else:
        for i in range(n_classifiers):
            class_index_list.append(np.where(coverage_matrix[i])[0])
    return class_index_list


def shrink_to_coverage(decision_tensor, coverage):
    """
    Shrink the given decision tensor to decision outputs according to the given coverage.
    Assumption: the normal class is covered by each classifier at index `0`.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of crisp multilabel decision outputs by different classifiers per sample.
    :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a classifier,
            which is identified by the positional index of the respective list.
    :return: `list` of `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is
            classifier-specific due to the coverage.
    """
    decision_tensor = np.array(decision_tensor)
    decision_outputs = []
    for i in range(len(decision_tensor)):
        sdt = decision_tensor[i, :, coverage[i]].T
        # assign the normal class for samples with no assignment.
        sdt = intercept_normal_class(sdt, override=True)
        decision_outputs.append(sdt)
    return tensorize(decision_outputs)


def split_into_train_and_validation_data(decision_tensor, true_assignments, validation_size=0.5):
    """
    Split the decision outputs (tensor) from multiple classifiers as well as the true assignments randomly into train
    and validation datasets.

    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
            Tensor of decision outputs by different classifiers per sample.
    :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
            Matrix of true class assignments.
    :param validation_size: Proportion between `0` and `1` for the size of the validation data set.
    :return: `tuple` of
            (1) `numpy.array` of shape `(n_classifiers, n_samples', n_classes)`,
            (2) `numpy.array` of shape `(n_classifiers, n_samples')`,
            (3) `numpy.array` of shape `(n_classifiers, n_samples'', n_classes)`,
            (4) `numpy.array` of shape `(n_classifiers, n_samples'')`, with `n_samples'` as the number of training
            samples and `n_samples''` as the number of validation samples.
    """
    n_validation_samples = int(len(true_assignments) * validation_size)
    all_indices = np.arange(len(true_assignments))
    validation_indices = np.random.choice(all_indices, n_validation_samples, replace=False)
    mask = np.ones(len(all_indices), bool)
    mask[validation_indices] = 0
    train_indices = all_indices[mask]
    true_assignments_train = true_assignments[train_indices]
    true_assignments_validation = true_assignments[validation_indices]
    decision_tensor_train = []
    decision_tensor_validation = []

    for decision_matrix in decision_tensor:
        decision_tensor_train.append(decision_matrix[train_indices])
        decision_tensor_validation.append(decision_matrix[validation_indices])

    return tensorize(decision_tensor_train), \
           tensorize(true_assignments_train), \
           tensorize(decision_tensor_validation), \
           tensorize(true_assignments_validation)
