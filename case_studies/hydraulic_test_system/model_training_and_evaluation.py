import multiprocessing as mp
import numpy as np
import sklearn.decomposition
from sklearn import neighbors, tree, ensemble, neural_network
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pusion.evaluation.evaluation_metrics
from pusion.util import class_assignment_matrix_to_label_vector, tensorize, transform_label_tensor_to_class_assignment_tensor


def k_plus_1_cross_validation_PCA_multi_class(X_folds: np.ndarray, Y_folds: np.ndarray, parallelize: bool = False, continuous_out: bool = False):
    # list of classifiers to be used
    classifiers = [
        # kNN
        neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=10, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=12, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=12, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=15, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=15, algorithm='auto', leaf_size=30, metric='minkowski', p=2),

        # decision trees
        tree.DecisionTreeClassifier(max_depth=3, criterion='gini', splitter='best', min_samples_split=2,
                                    min_samples_leaf=1),
        tree.DecisionTreeClassifier(max_depth=3, criterion='entropy', splitter='best', min_samples_split=2,
                                    min_samples_leaf=1),
        tree.DecisionTreeClassifier(max_depth=None, criterion='gini', splitter='best', min_samples_split=2,
                                    min_samples_leaf=1),
        tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', splitter='best', min_samples_split=2,
                                    min_samples_leaf=1),

        # random forest
        # ensemble.RandomForestClassifier(max_depth=3, n_estimators=9, random_state=1),
        # ensemble.RandomForestClassifier(max_depth=3, n_estimators=9, random_state=1),
        # ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

        # feedforward NN
        # neural_network.MLPClassifier(max_iter=5000, random_state=1, hidden_layer_sizes=(20, 20, 20)),
        # neural_network.MLPClassifier(max_iter=5000, random_state=2, hidden_layer_sizes=(100, 50)),
        # neural_network.MLPClassifier(max_iter=5000, random_state=3, hidden_layer_sizes=(100, 50, 25)),

        # SVM

    ]

    num_folds = len(X_folds.keys())
    for i_current_test_fold in range(num_folds):
        current_X_test_fold = X_folds[i_current_test_fold]
        current_Y_test_fold = Y_folds[i_current_test_fold]
        # print("+ Test Fold: " + str(i_current_test_fold))

        for j_current_val_fold in range(num_folds):
            if j_current_val_fold != i_current_test_fold:
                current_X_val_fold = X_folds[j_current_val_fold]
                current_Y_val_fold = Y_folds[j_current_val_fold]
                X_training_folds_list = []
                Y_training_folds_list = []
                # print("++ Validation Fold: " + str(j_current_val_fold))
                stri = "["
                for k_current_test_fold in range(num_folds):
                    if k_current_test_fold != i_current_test_fold and k_current_test_fold != j_current_val_fold:
                        X_training_folds_list.append(X_folds[k_current_test_fold])
                        Y_training_folds_list.append(Y_folds[k_current_test_fold])
                        stri += str(k_current_test_fold) + " "

                X_training_folds = np.concatenate(X_training_folds_list, axis=0)
                Y_training_folds = np.concatenate(Y_training_folds_list, axis=0)

                #########################################################################
                # dimension reduction via PCA
                pca = sklearn.decomposition.PCA(n_components=10)
                pca.fit(X_training_folds)

                X_training_folds_pca = pca.transform(X_training_folds)
                current_X_val_fold_pca = pca.transform(current_X_val_fold)
                current_X_test_fold_pca = pca.transform(current_X_test_fold)
                #########################################################################

                # train the models
                stri += "]"
                print("+ Test Fold: " + str(i_current_test_fold) + " ++ Validation Fold: " + str(
                    j_current_val_fold) + " +++ Training Folds: " + stri)

                # start model training with original data here
                # Classifier training
                if parallelize:
                    # Parallelized classifier training
                    # Create a thread-safe queue for classifiers
                    queue = mp.Manager().Queue()
                    processes = []
                    for i, clf in enumerate(classifiers):
                        processes.append(
                            mp.Process(target=_process_fit_classifier,
                                       args=(i, clf, X_training_folds, Y_training_folds, queue)))
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
                        clf.fit(X_training_folds, Y_training_folds)


                # Classifier validation to generate combiner training data
                y_ensemble_valid = []
                if continuous_out:
                    for i, clf in enumerate(classifiers):
                        print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                        y_ensemble_valid.append(clf.predict_proba(current_X_val_fold))
                else:
                    for i, clf in enumerate(classifiers):
                        print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                        y_ensemble_valid.append(clf.predict(current_X_val_fold))

                # Classifier test
                y_ensemble_test = []
                if continuous_out:
                    for i, clf in enumerate(classifiers):
                        print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                        y_ensemble_test.append(clf.predict_proba(current_X_test_fold))
                else:
                    for i, clf in enumerate(classifiers):
                        print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                        y_ensemble_test.append(clf.predict(current_X_test_fold))

                # Transform to numpy tensors if possible
                y_ensemble_valid = np.array(y_ensemble_valid)  # TODO what do we do with this?
                y_ensemble_test = np.array(y_ensemble_test)

                evaluation_report = evaluate_multi_label_model(classifiers, current_Y_test_fold, y_ensemble_test,
                                                               continuous_out)

        print("---------------------")

    pass



def k_plus_1_cross_validation_multi_class(X_folds: np.ndarray, Y_folds: np.ndarray, parallelize: bool = False, continuous_out: bool = False):


    classifiers = [
        SVC(kernel="linear", probability=True, decision_function_shape='ovo'),
        SVC(kernel="poly", probability=True, decision_function_shape='ovo'),
        SVC(kernel="poly", probability=True, degree=7, decision_function_shape='ovo'),
        SVC(kernel="poly", probability=True, degree=9, decision_function_shape='ovo'),
        tree.DecisionTreeClassifier(max_depth=5, criterion='entropy', splitter='best', min_samples_split=2, min_samples_leaf=5),
        tree.DecisionTreeClassifier(max_depth=7, criterion='gini', splitter='best', min_samples_split=2, min_samples_leaf=5),
        tree.DecisionTreeClassifier(max_depth=10, criterion='entropy', splitter='best', min_samples_split=2,min_samples_leaf=5),
        tree.DecisionTreeClassifier(max_depth=15, criterion='entropy', splitter='best', min_samples_split=2,min_samples_leaf=5),
        tree.DecisionTreeClassifier(max_depth=None, criterion='gini', splitter='best', min_samples_split=2,min_samples_leaf=2),
        tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', splitter='best', min_samples_split=2,min_samples_leaf=5),
        neural_network.MLPClassifier(max_iter=1000, random_state=3, hidden_layer_sizes=(200, 200), activation='relu', solver='adam', batch_size=32, learning_rate_init=0.001,shuffle=True),
        neural_network.MLPClassifier(max_iter=1000, random_state=3, hidden_layer_sizes=(200, 200), activation='relu', solver='sgd', batch_size=32, learning_rate_init=0.01, learning_rate='adaptive', shuffle=True),
        neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=3, algorithm='auto', leaf_size=30, metric='minkowski', p=2),
        neighbors.KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        neighbors.KNeighborsClassifier(n_neighbors=7, algorithm='auto', leaf_size=30, metric='minkowski', p=1),
        ensemble.RandomForestClassifier(max_depth=None, n_estimators=200, max_features='auto', criterion='gini', min_samples_leaf=2),
        ensemble.RandomForestClassifier(max_depth=None, n_estimators=200, max_features='auto', criterion='entropy', min_samples_leaf=2),

    ]

    eval_metrics = [
        pusion.PerformanceMetric.MICRO_PRECISION,
        pusion.PerformanceMetric.MICRO_RECALL,
        pusion.PerformanceMetric.MACRO_F1_SCORE,
        pusion.PerformanceMetric.ACCURACY,
        pusion.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE,
        pusion.PerformanceMetric.ERROR_RATE
    ]

    results_dict = {}
    num_folds = len(X_folds.keys())
    for i_current_test_fold in range(num_folds):
        current_X_test_fold = X_folds[i_current_test_fold]
        current_Y_test_fold = Y_folds[i_current_test_fold]

        for j_current_val_fold in range(num_folds):
            if j_current_val_fold != i_current_test_fold:
                current_X_val_fold = X_folds[j_current_val_fold]
                current_Y_val_fold = Y_folds[j_current_val_fold]
                X_training_folds_list = []
                Y_training_folds_list = []
                stri = "["
                for k_current_test_fold in range(num_folds):
                    if k_current_test_fold != i_current_test_fold and k_current_test_fold != j_current_val_fold:
                        X_training_folds_list.append(X_folds[k_current_test_fold])
                        Y_training_folds_list.append(Y_folds[k_current_test_fold])
                        stri += str(k_current_test_fold) + " "


                X_training_folds = np.concatenate(X_training_folds_list, axis=0)
                Y_training_folds = np.concatenate(Y_training_folds_list, axis=0)

                n_classes = Y_training_folds.shape[1]
                Y_training_folds = class_assignment_matrix_to_label_vector(Y_training_folds)

                # train the models
                stri += "]"
                print("+ Test Fold: " + str(i_current_test_fold) + " ++ Validation Fold: " + str(
                    j_current_val_fold) + " +++ Training Folds: " + stri)

                # start model training with original data here
                # Classifier training
                if parallelize:
                    # Parallelized classifier training
                    # Create a thread-safe queue for classifiers
                    queue = mp.Manager().Queue()
                    processes = []
                    for i, clf in enumerate(classifiers):
                        processes.append(
                            #mp.Process(target=_process_fit_classifier, args=(i, clf, X_training_folds, np.argmax(Y_training_folds, axis=1), queue)))
                            mp.Process(target=_process_fit_classifier, args=(i, clf, X_training_folds, Y_training_folds, queue)))
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
                        clf.fit(X_training_folds, Y_training_folds)


                # Classifier validation to generate combiner training data
                y_ensemble_valid_continuous = []
                for i, clf in enumerate(classifiers):
                    print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                    y_ensemble_valid_continuous.append(clf.predict_proba(current_X_val_fold))

                y_ensemble_valid = []
                for i, clf in enumerate(classifiers):
                    print("Validate classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                    y_ensemble_valid.append(clf.predict(current_X_val_fold))

                # Classifier test
                y_ensemble_test_continuous = []
                for i, clf in enumerate(classifiers):
                    print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                    y_ensemble_test_continuous.append(clf.predict_proba(current_X_test_fold))

                y_ensemble_test = []
                for i, clf in enumerate(classifiers):
                    print("Test classifier: ", type(clf).__name__, " [" + str(i) + "] ...")
                    y_ensemble_test.append(clf.predict(current_X_test_fold))

                # Transform to numpy tensors if possible
                y_ensemble_valid = transform_label_tensor_to_class_assignment_tensor(y_ensemble_valid, n_classes)
                y_ensemble_test = transform_label_tensor_to_class_assignment_tensor(y_ensemble_test, n_classes)
                y_ensemble_valid = tensorize(y_ensemble_valid)
                y_ensemble_test = tensorize(y_ensemble_test)

                y_ensemble_test_continuous = np.array(y_ensemble_test_continuous)
                y_ensemble_valid_continuous = np.array(y_ensemble_valid_continuous)

                current_results = dict()
                current_results['y_ensemble_valid'] = y_ensemble_valid
                current_results['y_ensemble_test'] = y_ensemble_test
                current_results['y_ensemble_valid_continuous'] = y_ensemble_valid_continuous
                current_results['y_ensemble_test_continuous'] = y_ensemble_test_continuous
                current_results['i_current_test_fold'] = i_current_test_fold
                current_results['j_current_val_fold'] = j_current_val_fold

                evaluation_report = evaluate_multi_class_model(classifiers, current_Y_test_fold, y_ensemble_test, continuous_out=False, eval_metrics=eval_metrics)
                current_results['evaluation_report'] = evaluation_report

                evaluation_report_continuous = evalute_multi_class_model_continuous(classifiers, current_Y_test_fold, y_ensemble_test_continuous, continuous_out=True, eval_metrics=eval_metrics)

                fold_name = "train" + str(i_current_test_fold) + "_val" + str(j_current_val_fold)
                results_dict[fold_name] = current_results


        print("---------------------")

    results_dict['X_folds'] = X_folds
    results_dict['Y_folds'] = Y_folds
    results_dict['classifiers'] = classifiers
    results_dict['eval_metrics'] = eval_metrics

    return results_dict


def evaluate_multi_class_model(classifiers, y_test: np.ndarray, y_predictions: np.ndarray,
                               continuous_out: bool = False, eval_metrics=None):# -> dict[str, np.ndarray]:
    """
    :param continuous_out: "True" if predictions are continuous
    :param classifiers: the list of classifiers used
    :param y_test: the true labels
    :param y_predictions: the predicted classes
    :return: evaluation results of the models
    """

    # left out MEAN_MULTILABEL_ACCURACY because only works with multi_label (?)
    eval_metrics = [
        # pusion.PerformanceMetric.MICRO_PRECISION,
        # pusion.PerformanceMetric.MICRO_RECALL,
        # pusion.PerformanceMetric.MICRO_F1_SCORE,
        # pusion.PerformanceMetric.MICRO_F2_SCORE,
        # pusion.PerformanceMetric.MICRO_JACCARD_SCORE,
        # pusion.PerformanceMetric.MACRO_PRECISION,
        # pusion.PerformanceMetric.MACRO_RECALL,
        # pusion.PerformanceMetric.MACRO_F1_SCORE,
        # pusion.PerformanceMetric.MACRO_F2_SCORE,
        # pusion.PerformanceMetric.MACRO_JACCARD_SCORE,
        pusion.PerformanceMetric.ACCURACY,
        # pusion.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE
    ]


    eval_classifiers = pusion.Evaluation(*eval_metrics)
    eval_classifiers.set_instances(classifiers)
    #eval_classifiers.evaluate_classification_performance(y_test, y_predictions)
    eval_classifiers.evaluate(y_test, y_predictions)
    print(eval_classifiers.get_report())

    return eval_classifiers.get_report().records



def evalute_multi_class_model_continuous(classifiers, y_test: np.ndarray, y_predictions: np.ndarray,
                               continuous_out: bool = False, eval_metrics=None):# -> dict[str, np.ndarray]:
    """
    :param continuous_out: "True" if predictions are continuous
    :param classifiers: the list of classifiers used
    :param y_test: the true labels
    :param y_predictions: the predicted classes
    :return: evaluation results of the models
    """

    # left out MEAN_MULTILABEL_ACCURACY because only works with multi_label (?)
    eval_metrics = [
        # pusion.PerformanceMetric.MICRO_PRECISION,
        # pusion.PerformanceMetric.MICRO_RECALL,
        # pusion.PerformanceMetric.MICRO_F1_SCORE,
        # pusion.PerformanceMetric.MICRO_F2_SCORE,
        # pusion.PerformanceMetric.MICRO_JACCARD_SCORE,
        # pusion.PerformanceMetric.MACRO_PRECISION,
        # pusion.PerformanceMetric.MACRO_RECALL,
        # pusion.PerformanceMetric.MACRO_F1_SCORE,
        # pusion.PerformanceMetric.MACRO_F2_SCORE,
        # pusion.PerformanceMetric.MACRO_JACCARD_SCORE,
        #pusion.PerformanceMetric.ACCURACY,
        # pusion.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE
        pusion.PerformanceMetric.MEAN_CONFIDENCE,
        pusion.PerformanceMetric.MULTICLASS_BRIER_SCORE
    ]


    eval_classifiers = pusion.Evaluation(*eval_metrics)
    eval_classifiers.set_instances(classifiers)
    #eval_classifiers.evaluate_classification_performance(y_test, y_predictions)
    eval_classifiers.evaluate(y_test, y_predictions)
    print(eval_classifiers.get_report())

    return eval_classifiers.get_report().records



def _process_fit_classifier(index, classifier, x_train, y_train, queue):
    """
    Encapsulates classifier's fit procedure in order to be executable in a separate thread.

    :param index: Index that identifies the position within the thread safe queue.
    :param classifier: A classifier instance.
    :param x_train: Input data.
    :param y_train: Target values (i.e. labels or class assignments).
    :param queue: A thread safe queue for multiprocessing (e.g. `multiprocessing.managers.SyncManager.Queue`).
    """
    print("Train classifier: ", type(classifier).__name__, " [" + str(index) + "] ...")
    classifier.fit(x_train, y_train)
    queue.put((index, classifier))