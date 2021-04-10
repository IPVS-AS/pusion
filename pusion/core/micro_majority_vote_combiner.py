import numpy as np

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.constants import *


class MicroMajorityVoteCombiner(UtilityBasedCombiner):
    """
    The :class:`MicroMajorityVoteCombiner` (MIMV) is based on a variation of the general majority vote method.
    The fusion consists of a decision vector which results from the majority of assignments for each individual class.
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
        Combine decision outputs by MIMV across all classifiers per class (micro).
        Only crisp classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp label assignments obtained by MIMV. Axis 0 represents samples and
                axis 1 the class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_tensor = decision_tensor - .5
        decision_sum = np.sum(decision_tensor, axis=0)
        fused_decisions = (decision_sum > 0) * np.ones_like(decision_sum)  # or (decision_sum >= 0)?
        return fused_decisions


class CRMicroMajorityVoteCombiner(MicroMajorityVoteCombiner):
    """
    The :class:`CRMicroMajorityVoteCombiner` is a modification of :class:`MicroMajorityVoteCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use :meth:`combine` a coverage needs to
    be set first by the inherited :meth:`set_coverage` method.
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

    def combine(self, decision_outputs):
        """
        Combine decision outputs by MIMV across all classifiers per class (micro).
        Only crisp classification outputs are supported.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifier` classifiers and contains crisp decision outputs
                per sample.

        :return: A matrix (`numpy.array`) of crisp label assignments which are obtained by MIMV. Axis 0 represents
                samples and axis 1 all the class labels which are provided by the coverage.
        """
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
