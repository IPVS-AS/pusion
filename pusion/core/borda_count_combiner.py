from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class BordaCountCombiner(UtilityBasedCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'BC'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def combine(self, decision_tensor):
        """
        Combining decision outputs by the Borda Count (BC) method. BC establishes a ranking between label assignments
        for a sample. This ranking is implicitly given by continuous support outputs and is mapped to
        different amounts of votes (0 of L votes for the lowest support, and L-1 votes for the highest one).
        A class with the highest sum of these votes (borda counts) across all classifiers is considered as a winner
        for the final decision. Only continuous classification outputs are supported.

        :param decision_tensor: Tensor of continuous decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments {0,1} which are obtained by the Borda Count. Axis 0 represents
        samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor} input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0], dtype=int)

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            sort_indices = np.argsort(dp, axis=1)
            bc_dp = self.swap_index_and_values(sort_indices)
            # TODO use a relation for the multilabel problem?
            fused_decisions[i, np.argmax(np.sum(bc_dp, axis=0))] = 1
        return fused_decisions

    def swap_index_and_values(self, m):
        s = np.zeros_like(m)
        for i in range(len(m)):
            for j in range(len(m[i])):
                s[i, m[i, j]] = j
        return s


class CRBordaCountCombiner(BordaCountCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    SHORT_NAME = 'BC (CR)'

    def __init__(self):
        super().__init__()
        self.coverage = None

    def set_coverage(self, coverage):
        self.coverage = coverage

    def combine(self, decision_outputs):  # TODO doc, return includes all classes for the cr scenario
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        return super().combine(t_decision_outputs)

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifier = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        # use zeros to generate proper confusion matrices in the nb training phase
        t_decision_outputs = np.zeros((n_classifier, n_decisions, n_classes))
        for i in range(n_classifier):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
