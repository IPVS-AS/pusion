import numpy as np

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import decision_tensor_to_decision_profiles
from pusion.util.constants import *


class MacroMajorityVoteCombiner(UtilityBasedCombiner):
    """
    The :class:`MacroMajorityVoteCombiner` (MAMV) is based on a variation of the general majority vote method.
    The fusion consists of a decision vector which is given by the majority of the classifiers in the ensemble for a
    sample. MAMV does not consider outputs for each individual class (macro).
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'MAMV'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by majority voting across all classifiers considering the most common classification
        assignment (macro). Only crisp classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments obtained by MAMV. Axis 0 represents samples and
                axis 1 the class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            unique_decisions = np.unique(dp, axis=0, return_counts=True)
            decisions = unique_decisions[0]
            counts = unique_decisions[1]
            fused_decisions[i] = decisions[np.argmax(counts)]
        return fused_decisions
