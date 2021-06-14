from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class BordaCountCombiner(UtilityBasedCombiner):
    """
    The :class:`BordaCountCombiner` (BC) is a decision fusion method that establishes a ranking between label
    assignments for a sample. This ranking is implicitly given by continuous support outputs and is mapped to different
    amounts of votes (:math:`0` of :math:`L` votes for the lowest support, and :math:`L-1` votes for the highest one).
    A class with the highest sum of these votes (borda counts) across all classifiers is considered as a winner for the
    final decision.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'BC'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the Borda Count (BC) method. Firstly, the continuous classification is mapped to a
        ranking with respect to available classes for each sample. Those rankings are then summed up across all
        classifiers to establish total votes (borda counts) for each class in a sample. The class with the highest
        number of borda counts is considered as decision fusion. Only continuous classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which represents fused decisions.
                Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in
                ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0], dtype=int)

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            sort_indices = np.argsort(dp, axis=1)
            bc_dp = self.swap_index_and_values(sort_indices)
            fused_decisions[i, np.argmax(np.sum(bc_dp, axis=0))] = 1
        return fused_decisions

    def swap_index_and_values(self, m):
        s = np.zeros_like(m)
        for i in range(len(m)):
            for j in range(len(m[i])):
                s[i, m[i, j]] = j
        return s


class CRBordaCountCombiner(BordaCountCombiner):
    """
    The :class:`CRBordaCountCombiner` is a modification of :class:`BordaCountCombiner` that also supports
    complementary-redundant decision outputs. Therefore the input is transformed, such that all missing classification
    assignments are considered as `0`, respectively. To call :meth:`combine` a coverage needs to be set first
    by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    def __init__(self):
        super().__init__()

    def combine(self, decision_outputs):
        """
        Combine complementary-redundant decision outputs by the Borda Count (BC) method. Firstly, the continuous
        classification is mapped to a ranking with respect to available classes for each sample. Those rankings are then
        summed up across all classifiers to establish total votes (borda counts) for each class in a sample. The class
        with the highest number of borda counts is considered as decision fusion. Only continuous classification outputs
        are supported.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifiers` classifiers and contains continuous decision outputs
                per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which represents fused decisions.
                Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in
                ``decision_tensor`` input tensor.

        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        return super().combine(t_decision_outputs)

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifiers = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.zeros((n_classifiers, n_decisions, n_classes))
        for i in range(n_classifiers):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
