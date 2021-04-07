from pusion.core.combiner import TrainableCombiner, EvidenceBasedCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class NaiveBayesCombiner(TrainableCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT)
    ]

    SHORT_NAME = 'NB'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.confusion_matrices = None
        # Number of samples per class
        self.n_samples_per_class = None

    def set_evidence(self, evidence):  # TODO
        """
        :param evidence: List of accuracy measurement values of all classifiers which is aligned with decision_tensor
        on axis 0.
        """
        self.confusion_matrices = evidence
        self.n_samples_per_class = np.sum(evidence[0], axis=1)

    def train(self, decision_tensor, true_assignment):
        """
        Train the Naive Bayes combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        :param true_assignment: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        self.confusion_matrices = generate_multiclass_confusion_matrices(decision_tensor, true_assignment)
        # Number of samples of certain class (multiclass case)
        self.n_samples_per_class = np.sum(true_assignment, axis=0)
        # self.sum_of_samples_per_class = np.sum(confusion_matrices[0], axis=1)

    def combine(self, decision_tensor):
        """
        Combining decision outputs by using the Naive Bayes method according to Kuncheva (ref. [01]) and
        Titterington et al. (ref. [02]). Continuous decision outputs are converted to crisp multiclass
        predictions using the MAX rule. Combining requires a trained NaiveBayesCombiner.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        if self.confusion_matrices is None or self.n_samples_per_class is None:
            raise RuntimeError("Untrained model.")
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])
        for i in range(len(decision_profiles)):
            # transform to a multiclass decision profile
            dp = multiclass_predictions_to_decisions(decision_profiles[i])
            n_classes = np.shape(dp)[1]
            mu = np.zeros(n_classes)
            for j in range(n_classes):
                n_j = self.n_samples_per_class[j]
                mu[j] = n_j / len(decision_profiles)
                for k in range(np.shape(dp)[0]):
                    # mu[j] = mu[j] * self.confusion_matrices[k, j, np.argmax(dp[k])]
                    mu[j] = mu[j] * ((self.confusion_matrices[k, j, np.argmax(dp[k])] + 1/n_classes) / (n_j + 1))
            fused_decisions[i, np.argmax(mu)] = 1
        return fused_decisions


# TODO align confusion matrices
class CRNaiveBayesCombiner(NaiveBayesCombiner):  # TODO extend, extract (DT cr & DS cr...)?
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    SHORT_NAME = 'NB (CR)'

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
        # use zeros to generate proper confusion matrices in the nb training phase
        t_decision_outputs = np.zeros((n_classifier, n_decisions, n_classes))
        for i in range(n_classifier):
            t_decision_outputs[i, :, coverage[i]] = decision_outputs[i].T
        return t_decision_outputs
