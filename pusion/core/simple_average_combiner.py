import numpy as np

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import multilabel_predictions_to_decisions
from pusion.util.constants import *


class SimpleAverageCombiner(UtilityBasedCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'AVG'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def train(self, decision_tensor, true_assignments):
        pass

    def combine(self, decision_tensor):
        """
        Combining decision outputs by averaging the class support of each classifier in the given ensemble.

        :param decision_tensor: Tensor of continuous decision outputs  by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of continuous class supports [0,1] which are obtained by simple averaging.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        # return np.mean(decision_tensor, axis=0)  # TODO MKK?
        return multilabel_predictions_to_decisions(np.mean(decision_tensor, axis=0), .5)


class CRSimpleAverageCombiner(SimpleAverageCombiner):  # TODO extend...
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    SHORT_NAME = 'AVG (CR)'

    def __init__(self):
        super().__init__()
        self.coverage = None

    def set_coverage(self, coverage):
        self.coverage = coverage

    def combine(self, decision_outputs):  # TODO doc
        """
        Combining decision outputs by averaging the class support of each classifier in the given ensemble.

        :param decision_outputs: Tensor of continuous decision outputs  by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of continuous class supports [0,1] which are obtained by simple averaging.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        return multilabel_predictions_to_decisions(np.nanmean(t_decision_outputs, axis=0), .5)  # TODO MKK test

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifier = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.full((n_classifier, n_decisions, n_classes), np.nan)
        for i in range(n_classifier):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
