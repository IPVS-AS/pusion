from clunpy.transformer import *


class BehaviourKnowledgeSpaceCombiner:
    def __init__(self):
        self.unique_configs = None
        self.config_class_distribution = None

    def train(self, decision_outputs, true_assignment):
        """
        Train the Behaviour Knowledge Space model (BKS) by extracting the classification configuration from all
        classifiers and summarizing samples of each true class that leads to that configuration. This relationship is
        recorded in a lookup table. Only crisp classification outputs are supported.

        @param decision_outputs: Tensor of crisp  decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        @param true_assignment: Matrix of crisp label assignments {0,1} which is considered true for each sample during
        the training procedure (axis 0: samples; axis 1: classes).
        """
        if np.shape(decision_outputs)[1] != np.shape(true_assignment)[0]:
            raise TypeError("True assignment vector dimension does not match the number of samples.")
        configs = decision_outputs_to_configs(decision_outputs)
        unique_configs = np.unique(configs, axis=0)
        dax1 = np.shape(true_assignment)[1]
        dax0 = np.shape(unique_configs)[0]
        config_class_distribution = np.empty((dax0, dax1), dtype=int)

        for i in range(dax0):
            c = unique_configs[i]
            # Determine identical classification configurations for each of which
            # the number of samples is accumulated per true class assignment.
            b = np.array([np.all(c == configs, axis=1)] * dax1).transpose()
            config_class_distribution[i] = np.sum(true_assignment, axis=0, where=b)

        self.unique_configs = unique_configs
        self.config_class_distribution = np.array(config_class_distribution)

    def combine(self, decision_outputs):
        """
        Combining decision outputs by Behaviour Knowledge Space (BKS) method introduced by Huan [05] and
        Suen et al. [06]. This procedure involves looking up the most representative class for a given classification
        output regarding the behaviour of all classifiers in the ensamble. Only crisp classification outputs are
        supported. If a trained lookup entry for certain classification configuration is not present,
        no decision fusion can be made for the sample, which led to that configuration. In this case, the decision
        fusion is a zero element.

        @param decision_outputs: Tensor of crisp decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of crisp label assignments which are obtained by the best representative class for a certain
        classifier's behaviour per sample. Axis 0 represents samples and axis 1 the class labels which are aligned
        with axis 2 in C{decision_outputs} input tensor.
        """
        configs = decision_outputs_to_configs(decision_outputs)
        fused_decisions = np.zeros_like(decision_outputs[0])

        for i in range(len(configs)):
            # perform a lookup in unique_configs
            lookup = np.where(np.all(configs[i] == self.unique_configs, axis=1))[0]
            if lookup.size > 0:
                uc_index = lookup[0]
                # set the class decision according to the maximum sample numbers for this config
                fused_decisions[i, self.config_class_distribution[uc_index].argmax()] = 1
                # TODO Multilabel:
                # dist = self.config_class_distribution[uc_index]
                # fused_decisions[i, np.argwhere(dist == dist.max())] = 1
        return fused_decisions
