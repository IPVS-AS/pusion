from pusion.core.combiner import *
from pusion.util.constants import *
from pusion.util.transformer import *


class BehaviourKnowledgeSpaceCombiner(TrainableCombiner):
    """
    The :class:`BehaviourKnowledgeSpaceCombiner` (BKS) is adopted from the decision fusion method originally proposed by
    Huang, Suen et al. :footcite:`huang1993behavior`. BKS analyses the behaviour of multiple classifiers based on their
    classification outputs with respect to each available class.
    This behaviour is recorded by means of a lookup table, which is used for final combination of multiple
    classification outputs for a sample.

    .. footbibliography::

    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'BKS'

    def __init__(self):
        super().__init__()
        self.unique_configs = None
        self.config_class_distribution = None
        self.n_classes = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the Behaviour Knowledge Space model (BKS) by extracting the classification configuration from all
        classifiers and summarizing samples of each true class that leads to that configuration. This relationship is
        recorded in a lookup table. Only crisp classification outputs are supported.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which are considered true for each sample during
                the training procedure.
        """
        configs = decision_tensor_to_configs(decision_tensor)
        unique_configs = np.unique(configs, axis=0)
        self.n_classes = np.shape(true_assignments)[1]
        n_unique_configs = np.shape(unique_configs)[0]
        config_class_distribution = np.zeros((n_unique_configs, self.n_classes), dtype=int)

        for i in range(n_unique_configs):
            unique_config = unique_configs[i]
            # Determine identical classification configurations for each of which
            # the number of samples is accumulated per true class assignment.
            b = np.array([np.all(unique_config == configs, axis=1)] * self.n_classes).transpose()
            config_class_distribution[i] = np.sum(true_assignments, axis=0, where=b)

        self.unique_configs = unique_configs
        self.config_class_distribution = np.array(config_class_distribution)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the Behaviour Knowledge Space (BKS) method. This procedure involves looking up the
        most representative class for a given classification output regarding the behaviour of all classifiers in the
        ensemble. Only crisp classification outputs are supported. If a trained lookup entry is not present for a
        certain classification configuration, no decision fusion can be made for the sample, which led to that
        configuration. In this case, the decision fusion is a zero vector.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of crisp decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which are obtained by the best representative class
                for a certain classifier's behaviour per sample. Axis 0 represents samples and axis 1 the class labels
                which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        configs = decision_tensor_to_configs(decision_tensor)
        fused_decisions = np.zeros((len(decision_tensor[0]), self.n_classes))

        for i in range(len(configs)):
            # perform a lookup in unique_configs
            lookup = np.where(np.all(configs[i] == self.unique_configs, axis=1))[0]
            if lookup.size > 0:
                uc_index = lookup[0]
                # set the class decision according to the maximum sample numbers for this config
                fused_decisions[i, self.config_class_distribution[uc_index].argmax()] = 1
        return fused_decisions


class CRBehaviourKnowledgeSpaceCombiner(BehaviourKnowledgeSpaceCombiner):
    """
    The :class:`CRBehaviourKnowledgeSpaceCombiner` is a modification of :class:`BehaviourKnowledgeSpaceCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use methods :meth:`train` and
    :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    def __init__(self):
        super().__init__()

    def train(self, decision_outputs, true_assignments):
        """
        Train the Behaviour Knowledge Space model (BKS) by extracting the classification configuration from all
        classifiers and summarizing samples of each true class that leads to that configuration. This relationship is
        recorded in a lookup table. Only crisp classification outputs are supported.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifiers` classifiers and contains crisp decision outputs
                per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of crisp class assignments which is considered true for each sample during
                the training procedure.
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        super().train(t_decision_outputs, true_assignments)

    def combine(self, decision_outputs):
        """
        Combine decision outputs by the Behaviour Knowledge Space (BKS) method. This procedure involves looking up the
        most representative class for a given classification output regarding the behaviour of all classifiers in the
        ensemble. Only crisp classification outputs are supported. If a trained lookup entry is not present for a
        certain classification configuration, no decision fusion can be made for the sample, which led to that
        configuration. In this case, the decision fusion is a zero vector.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifiers` classifiers and contains crisp decision outputs
                per sample.

        :return: A matrix (`numpy.array`) of crisp class assignments which are obtained by the best representative class
                for a certain classifier's behaviour per sample. Axis 0 represents samples and axis 1 all the class
                labels which are provided by the coverage.

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
