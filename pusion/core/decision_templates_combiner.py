from pusion.core.combiner import TrainableCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class DecisionTemplatesCombiner(TrainableCombiner):
    """
    The :class:`DecisionTemplatesCombiner` (DT) is adopted from the decision fusion method originally proposed by
    Kuncheva :footcite:`kuncheva2014combining`. A decision template is the average matrix of all decision profiles,
    which correspond to samples of one specific class. A decision profile contains classification outputs from all
    classifiers for a sample in a row-wise fashion. The decision fusion is performed based on distance calculations
    between decision templates and the decision profile generated from the ensemble outputs.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'DT'

    def __init__(self):
        super().__init__()
        # all possible classification occurrences (class assignments) in training data
        self.distinct_class_assignments = None
        # decision templates according to kuncheva per distinct class assignment (aligned with
        # distinct_class_assignments)
        self.decision_templates = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the Decision Templates Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculating means of decision profiles (decision templates) for each true class.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        if np.shape(decision_tensor)[1] != np.shape(true_assignments)[0]:
            raise TypeError("True assignment vector dimension does not match the number of samples.")

        # represent outputs of multiple classifiers as a DP for each sample
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        self.distinct_class_assignments = np.unique(true_assignments, axis=0)

        self.decision_templates = np.zeros((len(self.distinct_class_assignments),
                                            np.shape(decision_profiles[0])[0],
                                            np.shape(decision_profiles[0])[1]))

        for i in range(len(self.distinct_class_assignments)):
            # calculate the mean decision profile (decision template) for each class assignment.
            label = self.distinct_class_assignments[i]
            label_indices = np.where(np.all(label == true_assignments, axis=1))[0]
            self.decision_templates[i] = np.average(decision_profiles[label_indices], axis=0)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by using the Decision Templates method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DecisionTemplatesCombiner`.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the minimum distance between decision profiles of ``decision_tensor`` and
                precalculated decision templates. Axis 0 represents samples and axis 1 the class assignments which
                are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])

        # Compute the euclidean distance between each DP (for a class assignment) and trained DT.
        # The class assignment associated with the DT with minimal distance to the DP
        # is considered as the fused decision.
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            dist = np.empty(len(self.decision_templates))
            for j in range(len(self.decision_templates)):
                dt = self.decision_templates[j]
                dist[j] = np.average((dp - dt)**2)
            min_dist_label = self.distinct_class_assignments[dist.argmin()]
            fused_decisions[i] = min_dist_label
        return fused_decisions

    def get_decision_templates(self):
        return self.decision_templates

    def get_distinct_labels(self):
        return self.distinct_class_assignments


class CRDecisionTemplatesCombiner(DecisionTemplatesCombiner):
    """
    The :class:`CRDecisionTemplatesCombiner` is a modification of :class:`DecisionTemplatesCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use methods :meth:`train` and
    :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    def __init__(self):
        super().__init__()

    def train(self, decision_outputs, true_assignments):
        """
        Train the Decision Templates Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculating means of decision profiles (decision templates) for each true class.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifiers` classifiers and contains either crisp or continuous
                decision outputs per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        super().train(t_decision_outputs, true_assignments)

    def combine(self, decision_outputs):
        """
        Combine decision outputs by using the Decision Templates method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`CRDecisionTemplatesCombiner`.

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
        t_decision_outputs = np.negative(np.ones((n_classifiers, n_decisions, n_classes)))
        for i in range(n_classifiers):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
