import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

from pusion.core.combiner import UtilityBasedCombiner
from pusion.util.transformer import decision_tensor_to_decision_profiles
from pusion.util.constants import *


class CosineSimilarityCombiner(UtilityBasedCombiner):
    """
    The :class:`CosineSimilarityCombiner` considers the classification assignments to :math:`\\ell` classes as vectors
    from an :math:`\\ell`-dimensional vector space. The normalized cosine-similarity measure between two vectors
    :math:`x` and :math:`y` is calculated as

    .. math::
            cos(x,y) = \\dfrac{x\\cdot y}{|x||y|}\\ .

    The cosine-similarity is calculated pairwise and accumulated for each classifier for one specific sample.
    The fusion is represented by a classifier which shows the most similar classification output to the output of all
    competing classifiers.
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

    def combine(self, decision_tensor):
        """
        Combine decision outputs with as an output that accommodates the highest cosine-similarity to the output of
        all competing classifiers. In other words, the best representative classification output among the others is
        selected according to the highest cumulative cosine-similarity. This method supports both, continuous and
        crisp classification outputs.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.


        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the highest cumulative cosine-similarity. Axis 0 represents samples and axis 1 the
                class labels which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)

        for i, dp in enumerate(decision_profiles):
            s = np.sum(cosine_similarity(dp), axis=0)
            fused_decisions[i] = dp[np.argmax(s)]
        return fused_decisions


class CRCosineSimilarity(CosineSimilarityCombiner):
    """
    The :class:`CRCosineSimilarity` is a modification of :class:`CosineSimilarityCombiner` that also supports
    complementary-redundant decision outputs. Therefore the input is transformed, such that all missing classification
    assignments are considered as `0`, respectively. To call :meth:`combine` a coverage needs to be set first
    by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    def __init__(self):
        super().__init__()

    def combine(self, decision_outputs):
        """
        Combine complementary-redundant decision outputs with as an output that accommodates the highest
        cosine-similarity to the output of all competing classifiers. In other words, the best representative
        classification output among the others is selected according to the highest cumulative cosine-similarity.
        This method supports both, continuous and crisp classification outputs.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage. Each matrix corresponds to
                one of `n_classifiers` classifiers and contains crisp or continuous decision outputs per sample.

        :return: A matrix (`numpy.array`) of crisp or continuous class assignments which represents fused decisions.
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
