from pusion.core.decision_templates_combiner import *


class DempsterShaferCombiner(TrainableCombiner):
    """
    The :class:`DempsterShaferCombiner` (DS) fuses decision outputs by means of the Dempster Shafer evidence theory
    referenced by Polikar :footcite:`polikar2006ensemble` and Ghosh et al. :footcite:`ghosh2011evaluation`.
    DS involves computing the `proximity` and `belief` values per classifier and class, depending on a sample.
    Then, the total class support is calculated using the Dempster's rule as the product of belief values across all
    classifiers to each class, respectively. The class with the highest product is considered as a fused decision.
    DS shares the same training procedure with the :class:`DecisionTemplatesCombiner`.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'DS'

    def __init__(self):
        TrainableCombiner.__init__(self)
        self.decision_templates = None
        self.distinct_labels = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the Dempster Shafer Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculations mean decision profiles (decision templates) for each true class assignment.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        dt_combiner = DecisionTemplatesCombiner()
        dt_combiner.train(decision_tensor, true_assignments)
        self.decision_templates = dt_combiner.get_decision_templates()
        self.distinct_labels = dt_combiner.get_distinct_labels()

    def combine(self, decision_tensor):
        """
        Combine decision outputs by using the Dempster Shafer method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DempsterShaferCombiner`.
        This procedure involves computing the proximity, the belief values, and the total class support using the
        Dempster's rule.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of either crisp or continuous class assignments which represents fused
                decisions obtained by the maximum class support. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        fused_decisions = np.zeros_like(decision_tensor[0])

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            n_label = len(self.decision_templates)
            n_classifiers = len(decision_tensor)
            
            # Compute proximity
            prox = np.empty((n_label, n_classifiers))  # Phi_{j,k}
            for j in range(n_label):
                dt = self.decision_templates[j]
                for k in range(n_classifiers):
                    d = 0.0
                    for j_ in range(n_label):
                        d = d + (1 + np.linalg.norm(self.decision_templates[j_][k] - dp[k]))**(-1)

                    prox[j, k] = (1 + np.linalg.norm(dt[k] - dp[k]))**(-1) / d

            # Compute belief
            bel = np.empty((n_label, n_classifiers))  # bel_{j,k}
            for j in range(n_label):
                for k in range(n_classifiers):
                    prod = 1.0
                    for j_ in range(n_label):
                        if j_ != j:
                            prod = prod * (1 - prox[j_, k])

                    bel[j, k] = prox[j, k] * prod / (1 - prox[j, k] * (1 - prod))

            # Compute support for each label (Dempster's rule)
            mu = np.zeros(n_label)
            for j in range(n_label):
                prod = 1.0
                for k in range(n_classifiers):
                    prod = prod * bel[j, k]
                mu[j] = prod

            # normalization
            mu = mu / np.sum(mu)
            fused_decisions[i] = self.distinct_labels[np.argmax(mu)]

        return fused_decisions


class CRDempsterShaferCombiner(DempsterShaferCombiner):
    """
    The :class:`CRDempsterShaferCombiner` is a modification of :class:`DempsterShaferCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed, such that all missing
    classification assignments are considered as a constant, respectively. To use methods :meth:`train` and
    :meth:`combine` a coverage needs to be set first by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
    ]

    def __init__(self):
        super().__init__()

    def train(self, decision_outputs, true_assignments):
        """
        Train the Dempster Shafer Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculations mean decision profiles (decision templates) for each true class assignment.

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
        Combine decision outputs by using the Dempster Shafer method.
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        :class:`DempsterShaferCombiner`.
        This procedure involves computing the proximity, the belief values, and the total class support using the
        Dempster's rule.

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
