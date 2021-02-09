import numpy as np


class MicroMajorityVoteCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs):
        """
        Combining decision outputs by majority voting across all classifiers per class (micro).
        Both continuous and crisp classification outputs are supported.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of crisp label assignments {0,1} which are obtained by Micro majority vote.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        decision_sum = np.sum(decision_outputs, axis=0)
        mv_indices = np.argmax(decision_sum, axis=1)
        fused_decisions = np.zeros_like(decision_sum)
        fused_decisions[np.arange(len(decision_sum)), mv_indices] = 1
        return fused_decisions
