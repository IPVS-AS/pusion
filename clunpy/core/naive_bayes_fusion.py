from clunpy.transformer import *


class NaiveBayesCombiner:
    def __init__(self):
        self.confusion_matrices = None
        # Number of samples per class
        self.number_samples = None

    def train(self, decision_outputs, true_assignment):
        """
        Train the Naive Bayes combiner model by precalculating confusion matrices from given decision outputs and
        true class assignments. Continuous decision outputs are converted into crisp multiclass assignments using
        the MAX rule.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @param true_assignment: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        self.confusion_matrices = generate_multiclass_confusion_matrices(decision_outputs, true_assignment)
        # Number of samples of certain class (multiclass case)
        self.number_samples = np.sum(true_assignment, axis=0)
        # self.sum_of_samples_per_class = np.sum(confusion_matrices[0], axis=1)

    def combine(self, decision_outputs):
        """
        Combining decision outputs by using the Naive Bayes method according to Kuncheva (ref. [01]) and
        Titterington et al. (ref. [02]). Continuous decision outputs are converted to crisp multiclass
        predictions using the MAX rule. Combining requires a trained NaiveBayesCombiner.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of crisp label assignments {0,1} which are obtained by the maximum weighted class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        if self.confusion_matrices is None or self.number_samples is None:
            raise RuntimeError("Untrained model.")
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        fused_decisions = np.zeros_like(decision_outputs[0])
        for i in range(len(decision_profiles)):
            # transform to a multiclass decision profile
            dp = multiclass_predictions_to_decisions(decision_profiles[i])
            c = np.shape(dp)[1]
            mu = np.zeros(c)
            for j in range(c):
                n_j = self.number_samples[j]
                mu[j] = n_j / len(decision_profiles)
                for k in range(np.shape(dp)[0]):
                    # mu[j] = mu[j] * self.confusion_matrices[k, j, np.argmax(dp[k])]
                    mu[j] = mu[j] * ((self.confusion_matrices[k, j, np.argmax(dp[k])] + 1/c) / (n_j + 1))
            fused_decisions[i, np.argmax(mu)] = 1
        return fused_decisions

