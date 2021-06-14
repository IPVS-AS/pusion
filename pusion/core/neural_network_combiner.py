from sklearn.neural_network import MLPClassifier

from pusion.core.combiner import TrainableCombiner
from pusion.util.transformer import *
from pusion.util.constants import *


class NeuralNetworkCombiner(TrainableCombiner):
    """
    The :class:`NeuralNetworkCombiner` (NN) is a learning and classifier-based combiner that converts multiple decision
    outputs into new features, which in turn are used to train this combiner. The NN includes three hidden layers and a
    dynamic number of neurons per layer, which is given by (`n_classifiers * n_classes`).
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'NN'

    N_LAYERS = 3

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.classifier = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the NN combiner by fitting the Neural Network model with given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported.
        This procedure transforms decision outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        hidden_layer_sizes = (true_assignments.shape[1]*len(decision_tensor),) * self.N_LAYERS
        self.classifier = MLPClassifier(max_iter=5000, hidden_layer_sizes=hidden_layer_sizes)

        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        self.classifier.fit(featured_decisions, true_assignments)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the trained Neural Network model. Both continuous and crisp classification outputs
        are supported. Combining requires a trained :class:`NeuralNetworkCombiner`. This procedure transforms decision
        outputs into a new feature space.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by NN. Axis 0 represents samples and axis 1 the class assignments which are aligned
                with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # transfer decisions into a new feature space
        featured_decisions = decision_profiles.reshape((decision_profiles.shape[0], -1))

        # return self.classifier.predict_proba(featured_decisions)  # continuous output
        return self.classifier.predict(featured_decisions)


class CRNeuralNetworkCombiner(NeuralNetworkCombiner):
    """
    The :class:`CRNeuralNetworkCombiner` is a modification of :class:`NeuralNetworkCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use methods :meth:`train` and
    :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    def __init__(self):
        super().__init__()

    def train(self, decision_outputs, true_assignments):
        """
        Train the NN combiner by fitting the Neural Network model with given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported.
        This procedure transforms decision outputs into a new feature space.

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
        Combine decision outputs by the trained Neural Network model. Both continuous and crisp classification outputs
        are supported. Combining requires a trained :class:`NeuralNetworkCombiner`. This procedure transforms decision
        outputs into a new feature space.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage. Each matrix corresponds to
                one of `n_classifiers` classifiers and contains crisp or continuous decision outputs per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by NN. Axis 0 represents samples and axis 1 the class assignments which are aligned
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
