import numpy as np

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.constants import *


class MicroMajorityVoteCombiner(UtilityBasedCombiner):
    """
    MicroMajorityVoteCombiner
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'MIMV'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def train(self, decision_tensor, true_assignments):
        pass

    def combine(self, decision_tensor):
        """
        Combining decision outputs by majority voting across all classifiers per class (micro).
        Both continuous and crisp classification outputs are supported.

        :param decision_tensor: Tensor of crisp  decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments {0,1} which are obtained by Micro majority vote.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        decision_tensor = decision_tensor - .5
        decision_sum = np.sum(decision_tensor, axis=0)
        fused_decisions = (decision_sum > 0) * np.ones_like(decision_sum)  # or (decision_sum >= 0)?
        return fused_decisions


class CRMicroMajorityVoteCombiner(MicroMajorityVoteCombiner):  # TODO extend..., extract method
    """
    CRMicroMajorityVoteCombiner
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    SHORT_NAME = 'MIMV (CR)'

    def __init__(self):
        super().__init__()
        self.coverage = None

    def set_coverage(self, coverage):
        self.coverage = coverage

    def combine(self, decision_outputs):  # TODO doc
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        return super().combine(t_decision_outputs)

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifier = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.full((n_classifier, n_decisions, n_classes), .5)
        for i in range(n_classifier):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
