from pusion.core.combiner import TrainableCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class MaximumLikelihoodCombiner(TrainableCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'MLE'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.unique_assignments = None
        self.mu = []
        self.sigma = []

    def train(self, decision_tensor, true_assignments):
        """
        Train the Maximum Likelihood combiner model by calculating the parameters of gaussian normal distribution
        (i.e. means and variances) from the given decision outputs and true class assignments.

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

        # extract all occurring classifications
        self.unique_assignments, unique_inv_indices = np.unique(true_assignments, axis=0, return_inverse=True)

        # calculate the parameters for the multivariate normal distribution
        for i in range(len(self.unique_assignments)):
            xc = featured_decisions[np.where(unique_inv_indices == i)]  # X_train(class)
            mu = np.mean(xc, axis=0)
            sigma = 1 / len(xc) * np.sum(((xc - mu)*(xc - mu)).sum(1))
            if sigma == .0:
                sigma = .000001  # Add small perturbation in order to enable class conditional density
            self.mu.append(mu)
            self.sigma.append(sigma)

    def combine(self, decision_tensor):
        """
        Combining decision outputs by the Maximum Likelihood method.
        Both continuous and crisp classification outputs are supported.
        Combining requires a trained MaximumLikelihoodCombiner.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        :return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))  # TODO  MKK only
        # featured_decisions = np.sum(decision_profiles, axis=1)      # MLK
        # n = 1 / np.sum(featured_decisions, axis=1)                  # MLK
        # featured_decisions = featured_decisions * n[:, np.newaxis]  # MLK

        for i in range(len(featured_decisions)):
            x = featured_decisions[i]
            likelihoods = np.ones(len(self.unique_assignments))
            for j in range(len(self.unique_assignments)):
                for k in range(len(x)):
                    # calculate class conditional density for each dimension
                    exp = (x - self.mu[j])/self.sigma[j]
                    likelihoods[j] = likelihoods[j] * 1/(np.sqrt(2*np.pi) * self.sigma[j]) * np.e**(-.5*(exp*exp).sum())
            fused_decisions[i] = self.unique_assignments[np.argmax(likelihoods)]
        return fused_decisions


# TODO eval.

class CRMaximumLikelihoodCombiner(MaximumLikelihoodCombiner):  # TODO extend, extract (DT cr, DS cr, MLE cr)?
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    SHORT_NAME = 'MLE (CR)'

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
