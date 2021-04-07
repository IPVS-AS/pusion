from pusion.core.combiner import *
from pusion.util.constants import *
from pusion.util.transformer import *


class BehaviourKnowledgeSpaceCombiner(TrainableCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
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

        :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
                Tensor of crisp decision outputs :math:`\\{0,1\\}` by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments :math:`\\{0,1\\}` which is considered true for each sample during
                the training procedure.
        """
        # TODO disable for cr
        # if decision_tensor.shape[1] != true_assignments.shape[0]:
        #     raise TypeError("True assignment vector dimension does not match the number of samples.")
        configs = decision_tensor_to_configs(decision_tensor)
        unique_configs = np.unique(configs, axis=0)
        self.n_classes = np.shape(true_assignments)[1]
        n_unique_configs = np.shape(unique_configs)[0]
        # config_class_distribution = np.empty((n_unique_configs, self.n_classes), dtype=int)
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
        Combining decision outputs by Behaviour Knowledge Space (BKS) method introduced by Huan [05] and
        Suen et al. [06]. This procedure involves looking up the most representative class for a given classification
        output regarding the behaviour of all classifiers in the ensemble. Only crisp classification outputs are
        supported. If a trained lookup entry for certain classification configuration is not present,
        no decision fusion can be made for the sample, which led to that configuration. In this case, the decision
        fusion is a zero element.

        :param decision_tensor: Tensor of crisp decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments which are obtained by the best representative class for a certain
        classifier's behaviour per sample. Axis 0 represents samples and axis 1 the class labels which are aligned
        with axis 2 in C{decision_tensor} input tensor.
        """
        configs = decision_tensor_to_configs(decision_tensor)
        # fused_decisions = np.zeros_like(decision_tensor[0])  # TODO delete
        fused_decisions = np.zeros((len(decision_tensor[0]), self.n_classes))

        for i in range(len(configs)):
            # perform a lookup in unique_configs
            lookup = np.where(np.all(configs[i] == self.unique_configs, axis=1))[0]
            if lookup.size > 0:
                uc_index = lookup[0]
                # set the class decision according to the maximum sample numbers for this config
                fused_decisions[i, self.config_class_distribution[uc_index].argmax()] = 1
                # TODO Multilabel:
                # dist = self.config_class_distribution[uc_index]
                # fused_decisions[i, np.argwhere(dist == dist.max())] = 1
        return fused_decisions


class CRBehaviourKnowledgeSpaceCombiner(BehaviourKnowledgeSpaceCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    SHORT_NAME = 'BKS (CR)'

    def __init__(self):
        super().__init__()
        self.coverage = None

    def set_coverage(self, coverage):  # TODO delete, superclass Combiner contains the method and the attribute.
        self.coverage = coverage

    def train(self, decision_outputs, true_assignments):  # TODO update doc
        """
        Train the Behaviour Knowledge Space model (BKS) by extracting the classification configuration from all
        classifiers and summarizing samples of each true class that leads to that configuration. This relationship is
        recorded in a lookup table. Only crisp classification outputs are supported.

        :param decision_outputs: Tensor of crisp  decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :param true_assignments: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        super().train(t_decision_outputs, true_assignments)

    def combine(self, decision_outputs):  # TODO update doc
        """
        Combining decision outputs by Behaviour Knowledge Space (BKS) method introduced by Huan [05] and
        Suen et al. [06]. This procedure involves looking up the most representative class for a given classification
        output regarding the behaviour of all classifiers in the ensemble. Only crisp classification outputs are
        supported. If a trained lookup entry for certain classification configuration is not present,
        no decision fusion can be made for the sample, which led to that configuration. In this case, the decision
        fusion is a zero element.

        :param decision_outputs: Tensor of crisp decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments which are obtained by the best representative class for a certain
        classifier's behaviour per sample. Axis 0 represents samples and axis 1 the class labels which are aligned
        with axis 2 in C{decision_tensor} input tensor.
        """
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        return super().combine(t_decision_outputs)

    @staticmethod
    def __transform_to_uniform_decision_tensor(decision_outputs, coverage):
        n_classifier = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.negative(np.ones((n_classifier, n_decisions, n_classes)))
        for i in range(n_classifier):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
