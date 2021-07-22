from pusion.core.combiner import TrainableCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class MaximumLikelihoodCombiner(TrainableCombiner):
    """
    The :class:`MaximumLikelihoodCombiner` (MLE) is a combiner that estimates the parameters :math:`\\mu` (sample means)
    and :math:`\\sigma` (sample variances) of the Gaussian probability density function for each class :math:`\\omega`.
    Multiple decision outputs for a sample are converted into a new feature space.

    The fusion is performed by evaluating the class conditional density

    .. math::
        p(x|\\omega) = \\frac{1}{\\sigma \\sqrt{2 \\pi}}
            exp\\left({-\\frac{1}{2}\\left(\\frac{x-\\mu}{\\sigma}\\right)^2}\\right).

    of a transformed sample :math:`x` for each available class :math:`\\omega`, respectively. The class with the highest
    likelihood is considered as winner and thus forms the decision fusion.
    """

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
        Both continuous and crisp classification outputs are supported. This procedure transforms decision outputs
        into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        # extract all occurring classifications
        self.unique_assignments, unique_inv_indices = np.unique(true_assignments, axis=0, return_inverse=True)

        # calculate the parameters for the multivariate normal distribution
        for i in range(len(self.unique_assignments)):
            xc = featured_decisions[np.where(unique_inv_indices == i)]  # X_train(class)
            mu = np.mean(xc, axis=0)
            sigma = np.std(xc, axis=0)
            sigma[sigma == 0] = 0.00001  # Add a small perturbation in order to enable class conditional density

            self.mu.append(mu)
            self.sigma.append(sigma)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the Maximum Likelihood method. This procedure involves evaluating the class
        conditional density as described above. Both continuous and crisp classification outputs are supported.
        Combining requires a trained :class:`MaximumLikelihoodCombiner`.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by MLE. Axis 0 represents samples and axis 1 the class assignments which are aligned
                with axis 2 in ``decision_tensor`` input tensor.
        """
        fused_decisions = np.zeros_like(decision_tensor[0])
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        for i in range(len(featured_decisions)):
            x = featured_decisions[i]
            likelihoods = np.ones(len(self.unique_assignments))
            for j in range(len(self.unique_assignments)):
                for k in range(len(x)):
                    # calculate class conditional density for each dimension
                    exp = (x[k] - self.mu[j][k])/self.sigma[j][k]
                    likelihoods[j] = likelihoods[j] * 1/(np.sqrt(2*np.pi) * self.sigma[j][k]) * np.e**(-.5*(exp**2))
            fused_decisions[i] = self.unique_assignments[np.argmax(likelihoods)]
        return fused_decisions


class CRMaximumLikelihoodCombiner(MaximumLikelihoodCombiner):
    """
    The :class:`CRMaximumLikelihoodCombiner` is a modification of :class:`MaximumLikelihoodCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use methods :meth:`train` and
    :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        # (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT), # performance issues
        # (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT), # performance issues
    ]

    def __init__(self):
        super().__init__()

    def train(self, decision_outputs, true_assignments):
        """
        Train the Maximum Likelihood combiner model by calculating the parameters of gaussian normal distribution
        (i.e. means and variances) from the given decision outputs and true class assignments.
        Both continuous and crisp classification outputs are supported. This procedure transforms decision outputs
        into a new feature space.

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
        Combine decision outputs by the Maximum Likelihood method. This procedure involves evaluating the class
        conditional density as described above. Both continuous and crisp classification outputs are supported.
        Combining requires a trained :class:`MaximumLikelihoodCombiner`.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage. Each matrix corresponds to
                one of `n_classifiers` classifiers and contains crisp or continuous decision outputs per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by MLE. Axis 0 represents samples and axis 1 the class assignments which are aligned
                with axis 2 in ``decision_tensor`` input tensor.
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
