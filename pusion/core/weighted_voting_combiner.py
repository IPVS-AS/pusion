from pusion.core.combiner import TrainableCombiner, EvidenceBasedCombiner
from pusion.util.transformer import *
from sklearn.metrics import accuracy_score
from pusion.util.constants import *


class WeightedVotingCombiner(TrainableCombiner, EvidenceBasedCombiner):
    """
    The :class:`WeightedVotingCombiner` (WV) is a weighted voting schema adopted from Kuncheva (eq. 4.43)
    :footcite:`kuncheva2014combining`. Classifiers with better performance (i.e. accuracy) are given more
    weight contributing to final decisions. Nevertheless, if classifiers of high performance disagree on a sample,
    low performance classifiers may contribute to the final decision.

    .. footbibliography::
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'WV'

    def __init__(self):
        super().__init__()
        self.accuracy = None

    def train(self, decision_tensor, true_assignments):
        """
        Train the Weighted Voting combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of either crisp or continuous label assignments which are considered true for each sample during
                the training procedure.
        """
        cms = generate_multiclass_confusion_matrices(decision_tensor, true_assignments)
        self.accuracy = confusion_matrices_to_accuracy_vector(cms)

    def combine(self, decision_tensor):
        """
        Combine decision outputs by the weighted voting schema.
        Classifiers with better performance (i.e. accuracy) are given more authority over final decisions.
        Combining requires a trained :class:`WeightedVotingCombiner` or evidence set with ``set_evidence``.

        :param decision_tensor: `numpy.array` of shape `(n_classifier, n_samples, n_classes)`.
                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp label assignments which represents fused
                decisions obtained by the maximum weighted class support. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        if self.accuracy is None:
            raise TypeError("Accuracy is not set for this model as an evidence.")

        if np.shape(decision_tensor)[0] != np.shape(self.accuracy)[0]:
            raise TypeError("Accuracy vector dimension does not match the number of classifiers in the input tensor.")
        # convert decision_tensor to decision profiles for better handling
        decision_profiles = decision_tensor_to_decision_profiles(decision_tensor)
        # average all decision templates, i-th row contains decisions of i-th sample
        adp = np.array([np.average(dp, axis=0, weights=self.accuracy) for dp in decision_profiles])
        fused_decisions = np.zeros_like(adp)
        # find the maximum class support according to Kuncheva eq. (4.43)
        fused_decisions[np.arange(len(fused_decisions)), adp.argmax(1)] = 1
        return fused_decisions

    def set_evidence(self, evidence):
        """
        :param evidence: List of accuracy measurement values of all classifiers which is aligned with the
                ``decision_tensor`` on axis 0. Higher values indicate better accuracy. The accuracy is normalized to the
                [0,1]-interval.
        """
        self.accuracy = evidence


class CRWeightedVotingCombiner(WeightedVotingCombiner):
    """
    The :class:`CRWeightedVotingCombiner` is a modification of :class:`WeightedVotingCombiner` that
    also supports complementary-redundant decision outputs. Therefore the input is transformed to a unified
    tensor representation supporting undefined class assignments. The mean is calculated only for assignments which
    are defined. To call methods :meth:`train` and :meth:`combine`, a coverage needs to be set first
    by the inherited :meth:`set_coverage` method.
    """

    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    SHORT_NAME = 'WV (CR)'

    def __init__(self):
        super().__init__()
        self.accuracy = None
        self.coverage = None

    def set_coverage(self, coverage):
        self.coverage = coverage

    def train(self, decision_outputs, true_assignments):
        """
        Train the Weighted Voting combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifier` classifiers and contains crisp decision outputs
                per sample.

        :param true_assignments: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of crisp label assignments which is considered true for each sample during
                the training procedure.
        """
        self.accuracy = np.zeros(len(decision_outputs))
        for i in range(len(decision_outputs)):
            y_true = true_assignments[:, self.coverage[i]]
            y_pred = decision_outputs[i]
            # TODO accuracy per classifier / per classifier and class
            self.accuracy[i] = accuracy_score(y_true, y_pred)  # accuracy_score method?

    def combine(self, decision_outputs):
        """
        Combine decision outputs by the weighted voting schema.
        Classifiers with better performance (i.e. accuracy) are given more authority over final decisions.
        Combining requires a trained :class:`WeightedVotingCombiner` or evidence set with ``set_evidence``.

        :param decision_outputs: `list` of `numpy.array` matrices, each of shape `(n_samples, n_classes')`,
                where `n_classes'` is classifier-specific and described by the coverage.
                Each matrix corresponds to one of `n_classifier` classifiers and contains crisp decision outputs
                per sample.

        :return: A matrix (`numpy.array`) of crisp label assignments which are obtained by the best representative class
                for a certain classifier's behaviour per sample. Axis 0 represents samples and axis 1 all the class
                labels which are provided by the coverage.
        """
        if self.accuracy is None:
            raise TypeError("Accuracy is not set for this model as an evidence.")
        if len(decision_outputs) != len(self.accuracy):
            raise TypeError("Accuracy vector dimension does not match the number of classifiers in the input tensor.")

        # TODO begin - extract method?
        n_classifier = len(decision_outputs)
        n_decisions = len(decision_outputs[0])
        n_classes = len(np.unique(np.concatenate(self.coverage)))
        # tensor for transformed decision outputs
        t_decision_outputs = np.full((n_classifier, n_decisions, n_classes), np.nan)
        for i in range(n_classifier):
            t_decision_outputs[i, :, self.coverage[i]] = decision_outputs[i].T
        # TODO end
        # convert decision_tensor to decision profiles for better handling
        decision_profiles = decision_tensor_to_decision_profiles(t_decision_outputs)
        # use a masked array due to classifications which do not cover all classes
        masked_decision_profiles = np.ma.masked_array(decision_profiles, np.isnan(decision_profiles))
        # average all decision templates, i-th row contains decisions of i-th sample
        adp = np.array([np.ma.average(mdp, axis=0, weights=self.accuracy) for mdp in masked_decision_profiles])
        # adp = np.array([ad.filled(np.nan) for ad in adp])  # coverage test should avoid nan columns

        fused_decisions = np.zeros_like(adp)
        # find the maximum class support according to Kuncheva eq. (4.43)
        fused_decisions[np.arange(len(fused_decisions)), adp.argmax(1)] = 1
        return fused_decisions

    def set_evidence(self, evidence):
        """
        :param evidence: List of accuracy measurement values of all classifiers which is aligned with the
                ``decision_tensor`` on axis 0. Higher values indicate better accuracy. The accuracy is normalized to the
                [0,1]-interval.
        """
        self.accuracy = evidence
