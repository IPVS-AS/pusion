from sklearn.neural_network import MLPClassifier

from pusion.core.combiner import TrainableCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class NeuralNetworkCombiner(TrainableCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'NN'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.classifier = MLPClassifier(alpha=1, max_iter=5000)

    def train(self, decision_tensor, true_assignments):
        """
        Train the Neural Network combiner model from the given decision outputs and true class assignments.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        :param true_assignments: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))  # TODO  MKK only
        # featured_decisions = np.sum(decision_profiles, axis=1)      # MLK
        # n = 1 / np.sum(featured_decisions, axis=1)                  # MLK  # TODO dbz
        # featured_decisions = featured_decisions * n[:, np.newaxis]  # MLK
        self.classifier.fit(featured_decisions, true_assignments)

    def combine(self, decision_tensor):
        """
        Combining decision outputs by the Neural Network.
        Both continuous and crisp classification outputs are supported.
        Combining requires a trained NeuralNetworkCombiner.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))  # TODO  MKK only
        # featured_decisions = np.sum(decision_profiles, axis=1)      # MLK
        # n = 1 / np.sum(featured_decisions, axis=1)                  # MLK
        # featured_decisions = featured_decisions * n[:, np.newaxis]  # MLK

        return self.classifier.predict(featured_decisions)


class CRNeuralNetworkCombiner(NeuralNetworkCombiner):  # TODO extend, extract (DT cr, DS cr, MLE cr)?
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    SHORT_NAME = 'NN (CR)'

    def __init__(self):
        super().__init__()
        self.coverage = None

    def set_coverage(self, coverage):
        self.coverage = coverage

    # TODO doc class_ind. corr. to t_a, check class_indices cover? consistency of do between train and combine
    def train(self, decision_outputs, true_assignments):
        t_decision_outputs = self.__transform_to_uniform_decision_tensor(decision_outputs, self.coverage)
        super().train(t_decision_outputs, true_assignments)

    def combine(self, decision_outputs):  # TODO doc, return includes all classes for the cr scenario
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
