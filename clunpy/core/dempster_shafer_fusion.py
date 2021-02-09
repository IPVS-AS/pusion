from clunpy.core.decision_templates_fusion import *


class DempsterShaferCombiner:
    def __init__(self):
        self.decision_templates = None
        self.distinct_labels = None

    def train(self, decision_outputs, true_assignment):
        """
        Train the Dempster Shafer Combiner model by precalculating decision templates from given decision outputs and
        true class assignments. Both continuous and crisp classification outputs are supported. This procedure involves
        calculations mean decision profiles (decision templates) for each true label assignment.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @param true_assignment: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        dt_combiner = DecisionTemplatesCombiner()
        dt_combiner.train(decision_outputs, true_assignment)
        self.decision_templates = dt_combiner.get_decision_templates()
        self.distinct_labels = dt_combiner.get_distinct_labels()

    def combine(self, decision_outputs):
        """
        Combining decision outputs by using Dempster Shafer evidence theory referenced by Polikar [03] and
        Ghosh et al. [04]. Both continuous and crisp classification outputs are supported. Combining requires a trained
        DempsterShaferCombiner. This procedure involves computing the proximity, the belief values, and the total class
        support using the Dempster's rule.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of continuous or crisp label assignments which are obtained by the maximum class support.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        fused_decisions = np.zeros_like(decision_outputs[0])

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            n_label = len(self.decision_templates)
            n_classifiers = len(decision_outputs)
            
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

            # normalization - TODO use for multilabel output?
            mu = mu / np.sum(mu)
            fused_decisions[i] = self.distinct_labels[np.argmax(mu)]

        return fused_decisions
