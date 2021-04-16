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
        self.set_metrics(*argv)

    def evaluate_cr_combiners(self, true_assignments, decision_tensor, coverage):
        """
        Evaluate complementary-redundant decision outputs of multiple CR combiners with already set classification
        performance metrics.
        The evaluation results are averaged across complementary-redundant outputs obtained from each combiner
        for each coverage entry.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp label assignments which are considered true for the evaluation.
        :param decision_tensor: `numpy.array` of shape `(n_combiners, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different combiners per sample.
        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        if len(self.instances) != len(decision_tensor):
            raise TypeError("`decision_tensor` is not aligned with the number of instances.")
        performance_matrix = np.full((len(decision_tensor), len(self.metrics)), np.nan)
        for i in range(len(decision_tensor)):
            pm = self.evaluate_cr_combiner(true_assignments, decision_tensor[i], coverage)
            performance_matrix[i] = np.squeeze(pm)
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate_cr_combiner(self, true_assignments, decision_matrix, coverage):
        """
        Evaluate complementary-redundant decision outputs of a single CR combiner with already set classification
        performance metrics.
        The evaluation results are averaged across complementary-redundant outputs obtained from the ``decision_matrix``
        for each coverage entry.

        .. warning::

            This evaluation should be used only for CR combiners, in order to make a reasonable comparison between a
            CR ensemble (see ``evaluate_cr_ensemble``) and a CR combiner.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp label assignments which are considered true for the evaluation.
        :param decision_matrix: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp label assignments (fusion result) obtained by a CR combiner.
        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        cr_decision_outputs = []
        for i, cov in enumerate(coverage):
            cr_decision_outputs.append(decision_matrix[:, cov])
        performance_matrix = self.evaluate_cr_ensemble(true_assignments, cr_decision_outputs, coverage)
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate_cr_ensemble(self, true_assignments, cr_decision_outputs, coverage):
        """
        Evaluate complementary-redundant decision outputs with already set classification performance metrics.
        The evaluation results are averaged across complementary-redundant classifiers.

        .. warning::

            This evaluation is only applicable on complementary-redundant ensemble classifier outputs.

        :param true_assignments: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments which are considered true for the evaluation.
        :param cr_decision_outputs: `numpy.array` of shape `(n_classifier, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.
        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
        :return: `numpy.array` of shape `(n_instances, n_metrics)`. Performance matrix containing performance values
                for each set instance row-wise and each set performance metric column-wise.
        """
        self.__check()
        if len(cr_decision_outputs) != len(coverage):
            raise TypeError("`cr_decision_outputs` is not aligned to `coverage`.")

        performance_matrix = np.full((1, len(self.metrics)), np.nan)
        for i, metric in enumerate(self.metrics):
            score = 0.0
            for j, cr_do in enumerate(cr_decision_outputs):
                ta = intercept_normal_class(true_assignments[:, coverage[j]], override=True)
                do = intercept_normal_class(cr_do, override=True)
                score += metric(ta, do)
            avg_score = score / len(cr_decision_outputs)
            performance_matrix[0, i] = avg_score
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate(self, true_assignments, decision_tensor):
        """
        Evaluate the decision outputs with already set classification performance metrics.

        .. warning::

            This evaluation is only applicable on redundant multiclass or multilabel decision outputs.

        :param true_assignments: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments which are considered true for the evaluation.
        :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
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

    def get_report(self):
        """
        :return: A summary `Report` of performed evaluations including all involved instances and performance metrics.
        """
        return Report(np.around(self.performance_matrix, 3), self.instances, self.metrics)

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
        instance_indices = self.performance_matrix[:, metric_index]
        top_n_instance_indices = instance_indices.argsort()[-n:][::-1]
        return [(self.instances[i], self.performance_matrix[i, metric_index]) for i in top_n_instance_indices]

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

    def __check(self):
        if len(self.metrics) == 0:
            raise TypeError("No metrics. Use set_metrics(...) to set evaluation metrics before.")
        if self.instances is None:
            raise TypeError("No instances. Use set_instances(...) to set instances to be evaluated.")
