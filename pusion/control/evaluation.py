from pusion.model.report import Report
from pusion.util.generator import *


class Evaluation:
    def __init__(self, *argv):
        self.metrics = []
        self.instances = None
        self.performance_matrix = None
        self.set_metrics(*argv)

    def evaluate_cr_combiners(self, true_assignment, decision_tensor, coverage):
        self.__check()
        if len(self.instances) != len(decision_tensor):
            raise TypeError("`decision_tensor` is not aligned with the number of instances.")
        performance_matrix = np.full((len(decision_tensor), len(self.metrics)), np.nan)
        for i in range(len(decision_tensor)):
            pm = self.evaluate_cr_combiner(true_assignment, decision_tensor[i], coverage)
            performance_matrix[i] = np.squeeze(pm)
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate_cr_combiner(self, true_assignment, decision_matrix, coverage):
        self.__check()
        cr_decision_outputs = []
        for i, cov in enumerate(coverage):
            cr_decision_outputs.append(decision_matrix[:, cov])
        performance_matrix = self.evaluate_cr_ensemble(true_assignment, cr_decision_outputs, coverage)
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate_cr_ensemble(self, true_assignment, cr_decision_outputs, coverage):
        self.__check()
        if len(cr_decision_outputs) != len(coverage):
            raise TypeError("`cr_decision_outputs` is not aligned to `coverage`.")

        performance_matrix = np.full((1, len(self.metrics)), np.nan)
        for i, metric in enumerate(self.metrics):
            score = 0.0
            for j, cr_do in enumerate(cr_decision_outputs):
                ta = intercept_normal_class(true_assignment[:, coverage[j]], override=True)
                do = intercept_normal_class(cr_do, override=True)
                score += metric(ta, do)
            avg_score = score / len(cr_decision_outputs)
            performance_matrix[0, i] = avg_score
        self.performance_matrix = performance_matrix
        return performance_matrix

    def evaluate(self, true_assignment, decision_tensor):
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
                score = metric(true_assignment, decision_tensor[i])
                performance_matrix[i, j] = score
        self.performance_matrix = performance_matrix
        return performance_matrix

    def get_report(self):
        return Report(np.around(self.performance_matrix, 3), self.instances, self.metrics)

    def get_instances(self):
        return self.instances

    def get_metrics(self):
        return self.metrics

    def get_performance_matrix(self):
        return self.performance_matrix

    def get_top_n_instances(self, n=None, metric=None):
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
        self.metrics = []
        for metric in argv:
            if metric not in self.metrics:
                self.metrics.append(metric)

    def set_instances(self, instances):
        if type(instances) != list:
            instances = [instances]
        self.instances = instances

    def __check(self):
        if len(self.metrics) == 0:
            raise TypeError("No metrics. Use set_metrics(...) to set evaluation metrics before.")
        if self.instances is None:
            raise TypeError("No instances. Use set_instances(...) to set instances to be evaluated.")
