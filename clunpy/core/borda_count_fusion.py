from clunpy.transformer import *


class BordaCountCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs):
        """
        Combining decision outputs by the Borda Count (BC) method. BC establishes a ranking between label assignments
        for a sample. This ranking is implicitly given by continuous support outputs and is mapped to
        different amounts of votes (0 of L votes for the lowest support, and L-1 votes for the highest one).
        A class with the highest sum of these votes (borda counts) across all classifiers is considered as a winner
        for the final decision. Only continuous classification outputs are supported.

        @param decision_outputs: Tensor of continuous decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of crisp label assignments {0,1} which are obtained by the Borda Count. Axis 0 represents
        samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs} input tensor.
        """
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        fused_decisions = np.zeros_like(decision_outputs[0], dtype=int)

        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            sort_indices = np.argsort(dp, axis=1)
            bc_dp = self.swap_index_and_values(sort_indices)
            # TODO use a relation for the multilabel problem?
            fused_decisions[i, np.argmax(np.sum(bc_dp, axis=0))] = 1
        return fused_decisions

    def swap_index_and_values(self, m):
        s = np.zeros_like(m)
        for i in range(len(m)):
            for j in range(len(m[i])):
                s[i, m[i, j]] = j
        return s
