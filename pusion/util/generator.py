from sklearn.datasets import make_classification, make_multilabel_classification
from sklearn.model_selection import train_test_split

from pusion.evaluation.evaluation_metrics import *
from pusion.util.transformer import *


def generate_classification_coverage(n_classifier, n_classes, overlap, normal_class=True):
    """
        Generate random complementary redundant classification indices for each classifier 0..(n_classifier-1).
        The coverage is drawn from normal distribution for all classifiers.
        However, it is guaranteed that each classifier covers at least one class regardless of the distribution.
        :param n_classifier: Number of classifiers representing the classifier 0..(n_classifier-1).
        :param n_classes: Number of classes representing the class label 0..(n_classes-1).
        :param overlap: Indicator between 0 and 1 for overall classifier overlapping in terms of classes.
                            If 0, only complementary class indices are obtained.
                            If 1, the overlapping is fully redundant.
        :param normal_class: If True, a class for the normal state is included for all classifiers as class index 0.
        :return: List of sorted class label indices for each classifier.
    """
    if normal_class:
        n_classes = n_classes - 1
    coverage_matrix = np.zeros((n_classifier, n_classes), dtype=int)
    n_selected_classifier = int(np.interp(overlap, [0, 1], [np.ceil(n_classifier/n_classes), n_classifier]))
    while np.any(coverage_matrix.sum(axis=1) < 1):
        coverage_matrix = np.zeros((n_classifier, n_classes), dtype=int)
        for i in range(n_classes):
            selected_classifier = np.random.choice(np.arange(n_classifier), n_selected_classifier, replace=False)
            coverage_matrix[selected_classifier, i] = 1

    class_index_list = []
    if normal_class:
        for i in range(n_classifier):
            class_index_list.append(np.array([0, *np.where(coverage_matrix[i])[0] + 1]))
    else:
        for i in range(n_classifier):
            class_index_list.append(np.where(coverage_matrix[i])[0])
    return class_index_list


def generate_multiclass_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage=None):
    x, y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_redundant=0,
                               n_informative=n_classes,
                               n_clusters_per_class=1)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

    # Classifier training
    for i, clf in enumerate(classifiers):
        print("Train classifier: ", type(clf).__name__, "...")
        clf.fit(x_train, y_train)

    # Classifier validation to generate combiner training data
    y_ensemble_valid = []
    for i, clf in enumerate(classifiers):
        print("Validate classifier: ", type(clf).__name__, "...")
        y_ensemble_valid.append(clf.predict(x_valid))

    # Classifier test
    y_ensemble_test = []
    for i, clf in enumerate(classifiers):
        print("Test classifier: ", type(clf).__name__, "...")
        y_ensemble_test.append(clf.predict(x_test))

    # Transform to matrix representation
    y_ensemble_valid = transform_label_tensor_to_class_assignment_tensor(y_ensemble_valid, n_classes)
    y_valid = transform_label_vector_to_class_assignment_matrix(y_valid, n_classes)
    y_ensemble_test = transform_label_tensor_to_class_assignment_tensor(y_ensemble_test, n_classes)
    y_test = transform_label_vector_to_class_assignment_matrix(y_test, n_classes)

    # Transform to numpy tensors if possible
    y_ensemble_valid = decision_outputs_to_decision_tensor(y_ensemble_valid)
    y_ensemble_test = decision_outputs_to_decision_tensor(y_ensemble_test)

    if coverage:
        y_ensemble_valid = shrink_to_coverage(y_ensemble_valid, coverage)
        y_ensemble_test = shrink_to_coverage(y_ensemble_test, coverage)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def generate_multilabel_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage=None):
    x, y = make_multilabel_classification(n_samples=n_samples,
                                          n_classes=n_classes-1,  # -1, due to normal class
                                          n_labels=2,
                                          allow_unlabeled=True)

    # Adapt y to address the normal class representation in case of an unlabeled output
    y = intercept_normal_class(y)

    x_train, x_meta, y_train, y_meta = train_test_split(x, y, test_size=.5)
    x_valid, x_test, y_valid, y_test = train_test_split(x_meta, y_meta, test_size=.5)

    # Classifier training
    for i, clf in enumerate(classifiers):
        print("Train classifier: ", type(clf).__name__, "...")
        clf.fit(x_train, y_train)

    # Classifier validation to generate combiner training data
    y_ensemble_valid = []
    for i, clf in enumerate(classifiers):
        print("Validate classifier: ", type(clf).__name__, "...")
        y_pred = clf.predict(x_valid)
        y_ensemble_valid.append(intercept_normal_class(y_pred, override=True))

    # Classifier test
    y_ensemble_test = []
    for i, clf in enumerate(classifiers):
        print("Test classifier: ", type(clf).__name__, "...")
        y_pred = clf.predict(x_test)
        y_ensemble_test.append(intercept_normal_class(y_pred, override=True))

    if coverage:
        y_ensemble_valid = shrink_to_coverage(y_ensemble_valid, coverage)
        y_ensemble_test = shrink_to_coverage(y_ensemble_test, coverage)

    # Transform to numpy tensors if possible
    y_ensemble_valid = decision_outputs_to_decision_tensor(y_ensemble_valid)
    y_ensemble_test = decision_outputs_to_decision_tensor(y_ensemble_test)

    return y_ensemble_valid, y_valid, y_ensemble_test, y_test


def shrink_to_coverage(decision_tensor, coverage):
    # Assumption: the normal class is covered by each classifier.
    decision_tensor = np.array(decision_tensor)
    decision_outputs = []
    for i in range(len(decision_tensor)):
        sdt = decision_tensor[i, :, coverage[i]].T
        # assign the normal class for samples with no assignment.
        sdt = intercept_normal_class(sdt, override=True)
        decision_outputs.append(sdt)
    return decision_outputs_to_decision_tensor(decision_outputs)
