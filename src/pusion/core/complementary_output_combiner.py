import numpy as np


from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.constants import *


class ComplementaryOutputCombiner(UtilityBasedCombiner):
    """
    The :class:`ComplementaryOutputCombiner` combines fully complementary decision outputs by concatenating individual
    decisions across classes for each sample.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY)
    ]

    SHORT_NAME = 'COB'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_outputs):
        """
        Combine fully complementary decision outputs by concatenating individual decisions according to the coverage
        of all classifiers. Due to the nature of complementary class coverage, no fusion between redundant class
        assignments is required.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage. Each matrix corresponds to
                one of `n_classifiers` classifiers and contains crisp or continuous decision outputs per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the highest cumulative cosine-similarity. Axis 0 represents samples and axis 1 the
                class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        n_classes = np.sum([len(ca) for ca in self.coverage])
        fused_decisions = np.zeros_like((len(decision_outputs[0]), n_classes))

        for i, classifier_coverage in enumerate(self.coverage):
            for ci in classifier_coverage:
                fused_decisions[:, ci] = decision_outputs[i, :, ci]
        return fused_decisions
