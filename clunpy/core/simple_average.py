import numpy as np


class SimpleAverageCombiner:
    def __init__(self):
        pass

    def combine(self, decision_outputs):
        """
        Combining decision outputs by averaging the class support of each classifier in the given ensamble.

        @param decision_outputs: Tensor of continuous decision outputs  by different classifiers per sample
        (axis 0: classifier; axis 1: samples; axis 2: classes).
        @return: Matrix of continuous class supports [0,1] which are obtained by simple averaging.
        Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in C{decision_outputs}
        input tensor.
        """
        return np.mean(decision_outputs, axis=0)
