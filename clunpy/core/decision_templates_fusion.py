from clunpy.transformer import *


class DecisionTemplatesCombiner:
    def __init__(self):
        # all possible classification occurrences (label assignments) in training data
        self.distinct_labels = None
        # decision templates according to kuncheva per distinct label assignment (aligned with distinct_labels)
        self.decision_templates = None

    def train(self, decision_outputs, true_assignment):
        """
        Train the Decision Templates Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculations mean decision profiles (decision templates) for each true label assignment.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @param true_assignment: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        if np.shape(decision_outputs)[1] != np.shape(true_assignment)[0]:
            raise TypeError("True assignment vector dimension does not match the number of samples.")

        # represent outputs of multiple classifiers as a DP for each sample
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        self.distinct_labels = np.unique(true_assignment, axis=0)

        self.decision_templates = np.zeros((len(self.distinct_labels),
                                            np.shape(decision_profiles[0])[0],
                                            np.shape(decision_profiles[0])[1]))

        for i in range(len(self.distinct_labels)):
            # calculate the mean decision profile (decision template) for each label assignment.
            label = self.distinct_labels[i]
            label_indices = np.where(np.all(label == true_assignment, axis=1))[0]
            self.decision_templates[i] = np.average(decision_profiles[label_indices], axis=0)

    def combine(self, decision_outputs):
        """
        Combining decision outputs by using the Decision Templates method introduced by Kuncheva (ref. [01]).
        Both continuous and crisp classification outputs are supported. Combining requires a trained
        DecisionTemplatesCombiner.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of continuous or crisp label assignments which are obtained by the minimum distance between
        decision profiles of C{decision_outputs} and precalculated decision templates.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        fused_decisions = np.zeros_like(decision_outputs[0])

        # Compute the euclidean distance between each DP (for a label assignment) and trained DT.
        # The label assignment associated with the DT with minimal distance to the DP
        # is considered as the fused decision.
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            dist = np.empty(len(self.decision_templates))
            for j in range(len(self.decision_templates)):
                dt = self.decision_templates[j]
                dist[j] = np.average((dp - dt)**2)
            min_dist_label = self.distinct_labels[dist.argmin()]
            fused_decisions[i] = min_dist_label
        return fused_decisions

    def get_decision_templates(self):
        return self.decision_templates

    def get_distinct_labels(self):
        return self.distinct_labels
