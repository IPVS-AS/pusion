import numpy as np

from pusion.model.report import Report
from pusion.util.generator import *


class Evaluation:
    """
    :class:`Evaluation` provides methods for evaluating decision outputs (i.e. combiners and classifiers) with different
    problems and coverage types.

    :param argv: Performance metric functions.
    """
    def __init__(self, *argv):
        self.metrics = []
        self.instances = None
        self.performance_matrix = None
        self.runtime_matrix = None
        self.set_metrics(*argv)

    def evaluate(self, true_assignments, decision_tensor):
        """
        Evaluate the decision outputs with already set classification performance metrics.

        .. warning::

            This evaluation is only applicable on redundant multiclass or multilabel decision outputs.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for the evaluation.
        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        # generalize to decision_tensor if a matrix is given
        decision_tensor = np.array(decision_tensor)
        if decision_tensor.ndim < 3:
            decision_tensor = np.expand_dims(decision_tensor, axis=0)

        if len(self.instances) != len(decision_tensor):
            raise TypeError("`decision_tensor` is not aligned with the number of instances.")

        performance_matrix = np.full((len(decision_tensor), len(self.metrics)), np.nan)
        for i in range(len(decision_tensor)):
            for j in range(len(self.metrics)):
                metric = self.metrics[j]
                score = metric(true_assignments, decision_tensor[i])
                performance_matrix[i, j] = score
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate_cr_decision_outputs(self, true_assignments, decision_outputs, coverage=None):
        """
        Evaluate complementary-redundant decision outputs with already set classification performance metrics.
        The outputs of each classifier for each class is considered as a binary output and thus, the performance is
        calculated class-wise and averaged across all classes, which are covered by individual classifiers.

        .. note::

            This evaluation is applicable on complementary-redundant ensemble classifier outputs.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for the evaluation.
        :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.
        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
                If none set, the coverage for fully redundant classification is chosen by default.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        # generalize to 3-dimensional decision_outputs if a matrix is given
        if type(decision_outputs) == np.ndarray and decision_outputs.ndim < 3:
            decision_outputs = np.expand_dims(decision_outputs, axis=0)

        if coverage is None:
            coverage = [np.arange(true_assignments.shape[1], dtype=int) for _ in range(len(decision_outputs))]

        perf_matrix = np.full((1, len(self.metrics)), np.nan)
        for i, metric in enumerate(self.metrics):
            class_wise_mean_score = self.class_wise_mean_score(true_assignments, decision_outputs, coverage, metric)
            perf_matrix[0, i] = np.mean(class_wise_mean_score)

        self.performance_matrix = perf_matrix
        return perf_matrix

    def evaluate_cr_multi_combiner_decision_outputs(self, true_assignments, decision_tensor):
        """
        Evaluate decision outputs of multiple CR combiners with already set classification performance metrics.
        The evaluation is performed by :func:`evaluate_cr_decision_outputs` for each combiner.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for the evaluation.
        :param decision_tensor: `numpy.array` of shape `(n_combiners, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different combiners per sample.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        if len(self.instances) != len(decision_tensor):
            raise TypeError("`decision_tensor` is not aligned with the number of instances.")

        performance_matrix = np.full((len(decision_tensor), len(self.metrics)), np.nan)
        for i in range(len(decision_tensor)):
            dt = np.expand_dims(decision_tensor[i], axis=0)
            pm = self.evaluate_cr_decision_outputs(true_assignments, dt)
            performance_matrix[i] = np.squeeze(pm)
        self.performance_matrix = performance_matrix
        return performance_matrix

    def class_wise_mean_score(self, true_assignments, decision_outputs, coverage, metric):
        """
        Calculate the class-wise mean score with the given metric for the given classification outputs.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for the evaluation.
        :param decision_outputs: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.
        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
                If none set, the coverage for fully redundant classification is chosen by default.
        :param metric: The score metric.
        :return: `numpy.array` of shape `(n_classes,)`. The mean score per class across all classifiers.
        """
        self.__check()
        if len(decision_outputs) != len(coverage):
            raise TypeError("`decision_outputs` is not aligned to `coverage`.")

        true_assignments_per_coverage = [intercept_normal_class(true_assignments[:, cov], True) for cov in coverage]
        classifier_class_score_matrix = np.full((len(decision_outputs), true_assignments.shape[1]), np.nan)
        for i in range(len(decision_outputs)):
            for ci, j in enumerate(coverage[i]):
                decision_matrix = np.array(decision_outputs[i])
                pred_decision_vector = decision_matrix[:, ci]
                true_decision_vector = true_assignments_per_coverage[i][:, ci]
                classifier_class_score_matrix[i, j] = metric(true_decision_vector, pred_decision_vector)

        return np.nanmean(classifier_class_score_matrix, axis=0)

    def get_report(self):
        """
        :return: A summary `Report` of performed evaluations including all involved instances and performance metrics.
        """
        return Report(np.around(self.performance_matrix, 4), self.instances, self.metrics)

    def get_runtime_report(self):
        """
        :return: A summary `Report` of train and combine runtimes for all involved instances.
        """
        if self.runtime_matrix is not None:
            return Report(np.around(self.runtime_matrix, 4), self.instances, ['t_train (sec)', 't_comb (sec)'])
        raise TypeError("A runtime matrix is not set.")

    def get_instances(self):
        """
        :return: A `list` of instances (i.e. combiner or classifiers) been evaluated.
        """
        return self.instances

    def get_metrics(self):
        """
        :return: A `list` of performance metrics been used for evaluation.
        """
        return self.metrics

    def get_performance_matrix(self):
        """
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        return self.performance_matrix

    def get_runtime_matrix(self):
        """
        :return: `numpy.array` of shape `(n_instances, 2)`. Runtime matrix containing runtimes
                for each set instance row-wise. The column at index `0` describes train times and the column at index
                `1` describes combine times.
        """
        return self.runtime_matrix

    def get_top_n_instances(self, n=None, metric=None):
        """
        Retrieve top `n` best instances according to the given `metric` in a sorted order.

        :param n: `integer`. Number of instances to be retrieved. If unset, all instances are retrieved.
        :param metric: The metric all instances are sorted by. If unset, the first metric is used.
        :return: Evaluated top `n` instances.
        """
        self.__check()
        if self.performance_matrix is None:
            raise TypeError("No evaluation performed.")
        # set default parameter values
        if n is None:
            n = self.performance_matrix.shape[0]
        if metric is None:
            metric = self.metrics[0]

        metric_index = self.metrics.index(metric)
        performance_values = self.performance_matrix[:, metric_index]
        top_n_instance_indices = performance_values.argsort()[-n:][::-1]
        return [(self.instances[i], self.performance_matrix[i, metric_index]) for i in top_n_instance_indices]

    def get_top_instances(self, metric=None):
        """
        Retrieve best performing instances according to the given `metric`.
        Multiple instances may be returned having the identical best performance score.

        :param metric: The metric all instances were evaluated with. If unset, the first metric is used.
        :return: Evaluated top instances according to their performance.
        """
        top_n_instances = self.get_top_n_instances(metric=metric)
        top_score = None
        top_instances = []
        for i, t in enumerate(top_n_instances):
            if top_score is None:
                top_score = t[1]
                top_instances.append(t)
                continue
            if t[1] == top_instances[0][1]:
                top_instances.append(t)
        return top_instances

    def get_instance_performance_tuples(self, metric=None):
        """
        Retrieve (instance, performance) tuples created for to the given `metric`.

        :param metric: The metric all instances are evaluated by. If unset, the first set metric is used.
        :return: `list` of (instance, performance) tuples.
        """
        self.__check()
        if self.performance_matrix is None:
            raise TypeError("No evaluation performed.")
        # set default parameter values
        if metric is None:
            metric = self.metrics[0]

        metric_index = self.metrics.index(metric)
        return [(self.instances[i], self.performance_matrix[i, metric_index]) for i in range(len(self.instances))]

    def set_metrics(self, *argv):
        """
        :param argv: Performance metric functions.
        """
        self.metrics = []
        for metric in argv:
            if metric not in self.metrics:
                self.metrics.append(metric)

    def set_instances(self, instances):
        """
        :param instances: An instance or a `list` of instances to be evaluated, e.g. classifiers or combiners.
        """
        if type(instances) != list:
            instances = [instances]
        self.instances = instances

    def set_runtimes(self, runtimes):
        """
        :param runtimes: A `tuple` of two lists of tuples describing the train and combine runtimes respectively.
                Each runtime list is aligned with the list of set instances.
        """
        runtime_matrix = np.full((len(self.instances), 2), np.nan)
        for t in range(len(runtimes)):
            for i in range(len(runtimes[t])):
                runtime_matrix[runtimes[t][i][0], t] = runtimes[t][i][1]
        self.runtime_matrix = runtime_matrix

    def __check(self):
        if len(self.metrics) == 0:
            raise TypeError("No metrics. Use set_metrics(...) to set evaluation metrics before.")
        if self.instances is None:
            raise TypeError("No instances. Use set_instances(...) to set instances to be evaluated.")
