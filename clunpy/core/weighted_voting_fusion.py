from clunpy.transformer import *


class WeightedVotingCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs, accuracy):
        """
        Combining decision outputs by using the weighted voting schema according to Kuncheva (ref. [01]) (Eq. 4.43).
        Classifiers with better performance (i.e. accuracy) are given more authority over final decisions.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @param accuracy: List of accuracy measurement values of all classifiers which is aligned with decision_outputs
        on axis 0. Higher values indicate better accuracy. The accuracy is normalized to [0,1]-interval for weighting.
        @return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        if np.shape(decision_outputs)[0] != np.shape(accuracy)[0]:
            raise TypeError("Accuracy vector dimension does not match the number of classifiers in the input tensor.")
        # convert decision_outputs to decision profiles for better handling
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        # average all decision templates, i-th row contains decisions of i-th sample
        adp = np.array([np.average(dp, axis=0, weights=accuracy) for dp in decision_profiles])
        fused_decisions = np.zeros_like(adp)
        # find the maximum class support according to Kuncheva eq. (4.43)
        fused_decisions[np.arange(len(fused_decisions)), adp.argmax(1)] = 1
        return fused_decisions

    def combine_multilabel_using_mean_accuracy(self):
        pass

    def combine_multilabel_using_max_accuracy(self):
        pass

    def combine_multilabel_using_authority(self):
        pass
