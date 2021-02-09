import numpy as np

from clunpy.transformer import decision_outputs_to_decision_profiles


class MacroMajorityVoteCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs):
        """
        Combining decision outputs by majority voting across all classifiers considering the most common classification
        assignment (macro). Only crisp classification outputs are supported.

        @param decision_outputs: Tensor of crisp decision outputs by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of crisp label assignments {0,1} which are obtained by Macro majority vote.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        fused_decisions = np.zeros_like(decision_outputs[0])
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            unique_decisions = np.unique(dp, axis=0, return_counts=True)
            decisions = unique_decisions[0]
            counts = unique_decisions[1]
            fused_decisions[i] = decisions[np.argmax(counts)]
        return fused_decisions
