import time


def p_train(index, combiner, decision_tensor, true_assignment, queue):
    """
    Encapsulates combiner's training procedure in order to be executable in a separate thread.

    :param index: Index that identifies the position within the thread safe queue.
    :param combiner: A combiner instance.
    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
            Tensor of either crisp or continuous decision outputs by different classifiers per sample.
    :param true_assignment: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
    :param queue: A thread safe queue for multiprocessing (e.g. `multiprocessing.managers.SyncManager.Queue`).
    """
    t_begin = time.perf_counter()
    combiner.train(decision_tensor, true_assignment)
    t_elapsed = time.perf_counter() - t_begin
    queue.put((index, combiner, t_elapsed))


def p_combine(index, combiner, decision_tensor, queue):
    """
    Encapsulates combiner's combine procedure in order to be executable in a separate thread.

    :param index: Index that identifies the position within the thread safe queue.
    :param combiner: A combiner instance.
    :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
            `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
            due to the coverage.
            Tensor of either crisp or continuous decision outputs by different classifiers per sample.
    :param queue: A thread safe queue for multiprocessing (e.g. `multiprocessing.managers.SyncManager.Queue`).
    """
    t_begin = time.perf_counter()
    decision_matrix = combiner.combine(decision_tensor)
    t_elapsed = time.perf_counter() - t_begin
    queue.put((index, decision_matrix, t_elapsed))


def p_fit(index, classifier, x_train, y_train, queue):
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
