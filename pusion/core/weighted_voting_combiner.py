from pusion.core.combiner import TrainableCombiner, EvidenceBasedCombiner
from pusion.util.transformer import *
from sklearn.metrics import accuracy_score
from pusion.util.constants import *


class WeightedVotingCombiner(TrainableCombiner, EvidenceBasedCombiner):
    _SUPPORTED_PAC = [
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
    ]

    SHORT_NAME = 'WV'

    def __init__(self):
        super().__init__()
        self.accuracy = None

    def train(self, decision_tensor, true_assignment):
        cms = generate_multiclass_confusion_matrices(decision_tensor, true_assignment)
        self.accuracy = confusion_matrices_to_accuracy_vector(cms)

    def combine(self, decision_tensor):
        """
        Combining decision outputs by using the weighted voting schema according to Kuncheva (ref. [01]) (Eq. 4.43).
        Classifiers with better performance (i.e. accuracy) are given more authority over final decisions.

        :param decision_tensor: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).

        :return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_tensor}
        input tensor.
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
        :param evidence: List of accuracy measurement values of all classifiers which is aligned with decision_tensor
        on axis 0. Higher values indicate better accuracy. The accuracy is normalized to [0,1]-interval for weighting.
        """
        self.accuracy = evidence

    def combine_multilabel_using_mean_accuracy(self):
        pass

    def combine_multilabel_using_max_accuracy(self):
        pass

    def combine_multilabel_using_authority(self):
        pass


# TODO Evaluationsmöglichkeit 1: Accuracy pro Classifier
# TODO Evaluationsmöglichkeit 2: Accuracy pro Classifier und Klasse
class CRWeightedVotingCombiner(WeightedVotingCombiner):
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

    def set_evidence(self, evidence):
        """
        :param evidence: List of accuracy measurement values of all classifiers which is aligned with decision_tensor
        on axis 0. Higher values indicate better accuracy. The accuracy is normalized to [0,1]-interval for weighting.
        """
        self.accuracy = evidence

    def train(self, decision_outputs, true_assignments):
        self.accuracy = np.zeros(len(decision_outputs))
        for i in range(len(decision_outputs)):
            y_true = true_assignments[:, self.coverage[i]]
            y_pred = decision_outputs[i]
            self.accuracy[i] = accuracy_score(y_true, y_pred)  # accuracy_score method?

    def combine(self, decision_outputs):  # TODO doc, test coverage
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
