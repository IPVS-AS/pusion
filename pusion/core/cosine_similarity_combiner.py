import numpy as np

from scipy import spatial

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import decision_tensor_to_decision_profiles
from pusion.util.constants import *


class CosineSimilarityCombiner(UtilityBasedCombiner):
    """
    CosineSimilarityCombiner
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'COS'

    def __init__(self):
        UtilityBasedCombiner.__init__(self)

    def train(self, decision_tensor, true_assignments):
        pass

    def combine(self, decision_tensor):
        """
        Combining decision outputs with as an output that accommodates the highest cosine-similarity to the output of
        all competing classifiers. In other words, the best representative classification output among the others is
        selected according to the highest cumulative cosine-similarity. Supports both, continuous and crisp classifier
        outputs.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes) without zero elements.
        :return: Matrix of crisp label assignments {0,1} which are obtained by the highest cumulative cosine-similarity.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            accumulated_cos_sim = np.zeros(len(dp))
            for j in range(len(dp)):
                for k in range(len(dp)):
                    if j != k:
                        # Calculate the cosine distance (assumption: no zero elements)
                        accumulated_cos_sim[j] = accumulated_cos_sim[j] + (1 - spatial.distance.cosine(dp[j], dp[k]))
            fused_decisions[i] = dp[np.argmax(accumulated_cos_sim)]
        return fused_decisions


class CRCosineSimilarity(CosineSimilarityCombiner):
    """
    CRCosineSimilarity
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    SHORT_NAME = 'COS (CR)'

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
