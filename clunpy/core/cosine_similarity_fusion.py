import numpy as np

from scipy import spatial
from clunpy.transformer import decision_outputs_to_decision_profiles


class CosineSimilarityCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs):
        """
        Combining decision outputs with as an output that accommodates the highest cosine-similarity to the output of
        all competing classifiers. In other words, the best representative classification output among the others is
        selected according to the highest cumulative cosine-similarity. Supports both, continuous and crisp classifier
        outputs.

        @param decision_outputs: Tensor of either crisp or continuous decision outputs by different classifiers
        per sample (axis 0: classifier; axis 1: samples; axis 2: classes) without zero elements.
        @return: Matrix of crisp label assignments {0,1} which are obtained by the highest cumulative cosine-similarity.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        fused_decisions = np.zeros_like(decision_outputs[0])
        decision_profiles = decision_outputs_to_decision_profiles(decision_outputs)
        for i in range(len(decision_profiles)):
            dp = decision_profiles[i]
            accumulated_cos_sim = np.zeros(len(dp))
            for j in range(len(dp)):
                for k in range(len(dp)):
                    if j != k:
                        # Calculate the cosine distance (assumption: no zero elements)
                        accumulated_cos_sim[j] = accumulated_cos_sim[j] + (1 - spatial.distance.cosine(dp[j], dp[k]))
            fused_decisions[i] = dp[np.argmax(accumulated_cos_sim)]
        return fused_decisions
