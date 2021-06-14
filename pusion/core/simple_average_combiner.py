import numpy as np

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import multilabel_predictions_to_decisions
from pusion.util.constants import *


class SimpleAverageCombiner(UtilityBasedCombiner):
    """
    The :class:`SimpleAverageCombiner` (AVG) fuses decisions using the arithmetic mean rule.
    The mean is calculated between decision vectors obtained by multiple ensemble classifiers for a sample.
    The AVG combiner is unaware of the input problem (multiclass/multilabel) or the assignment type (crisp/continuous).
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'AVG'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by averaging the class support of each classifier in the given ensemble.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp assignments which represents fused
                decisions obtained by the AVG method. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        # return np.mean(decision_tensor, axis=0)  # continuous output
        return multilabel_predictions_to_decisions(np.mean(decision_tensor, axis=0), .5)


class CRSimpleAverageCombiner(SimpleAverageCombiner):
    """
    The :class:`CRSimpleAverageCombiner` is a modification of :class:`SimpleAverageCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed to a unified
    tensor representation supporting undefined class assignments. The mean is calculated only for assignments which
    are defined. To call :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    def __init__(self):
        super().__init__()

    def combine(self, decision_outputs):
        """
        Combine decision outputs by averaging the defined class support of each classifier in the given ensemble.
        Undefined class supports are excluded from averaging.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifiers` classifiers and contains either crisp or continuous
                decision outputs per sample.

        :return: A matrix (`numpy.array`) of crisp assignments which represents fused
                decisions obtained by the AVG method. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        # return np.nanmean(t_decision_outputs, axis=0)  # continuous output
        return multilabel_predictions_to_decisions(np.nanmean(t_decision_outputs, axis=0), .5)

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifiers = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.full((n_classifiers, n_decisions, n_classes), np.nan)
        for i in range(n_classifiers):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
