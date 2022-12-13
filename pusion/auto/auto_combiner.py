import pusion as p
from pusion.auto.detector import *
from pusion.auto.generic_combiner import GenericCombiner
from pusion.evaluation.evaluation_metrics import *
from pusion.util.generator import *


class AutoCombiner(GenericCombiner):
    """
    The `AutoCombiner` allows for automatic decision fusion using all methods provided by the framework, which are
    applicable to the given problem. The key feature of this combiner is the transparency in terms of it's outer
    behaviour. Based on the usage (i.e. method calls) and the automatically detected configuration,
    the `AutoCombiner` preselects all compatible methods from `pusion.core`. The main purpose is to retrieve fusion
    results obtained by the methods with the best performance without further user interaction.
    """

    _SUPPORTED_PAC = [
        (Problem.GENERIC, AssignmentType.GENERIC, CoverageType.GENERIC),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CRISP, CoverageType.COMPLEMENTARY_REDUNDANT),
        (Problem.MULTI_LABEL, AssignmentType.CONTINUOUS, CoverageType.COMPLEMENTARY_REDUNDANT)
    ]

    def __init__(self):
        super().__init__()
        self.selected_combiner = None
        self.validation_size = 0.5
        self.eval_metric = p.PerformanceMetric.ACCURACY # default evaluation metric for the AutoFusion approach

    def train(self, decision_tensor, true_assignments, **kwargs):
        """
        Train the AutoCombiner (AC) model. This method detects the configuration based on the ``decision_tensor`` and
        trains all trainable combiners that are applicable to this configuration.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """

        # Split into train and validation data.
        if 'test_data' in kwargs.keys():
            dt_train = decision_tensor
            ta_train = true_assignments
            dt_valid = kwargs['test_data']['dt_valid']
            ta_valid = kwargs['test_data']['ta_valid']
        else:
            dt_train, ta_train, dt_valid, ta_valid = split_into_train_and_validation_data(decision_tensor,
                                                                                          true_assignments,
                                                                                          self.validation_size)

        # set specified evaluation metric
        if 'eval_metric' in kwargs.keys():
            self.eval_metric = kwargs['eval_metric']

        # Encapsulated training phase.
        super().train(dt_train, ta_train)
        # Encapsulated evaluation phase.
        super().combine(dt_valid)
        # Todo save the decision fusion evaluation report within the AutoFusion procedure to be able to print the
        #  overview of the performance of the tested decision fusion methods
        performance_per_combiner = np.zeros(len(self.combiners))
        for i in range(len(self.combiners)):
            comb_res = self.multi_combiner_decision_tensor[i]
            # TODO add support for a combined evaluation metric a la Alejandro Villanuevas Paper in the 3d space
            #  --> add a weighted distance-based selection approach to select the best performing decision fusion algorithm, e. g. acc, fdr and prbabilistic prediction quality
            #  also further approaches, e. g. significance tests, cutomized cost functions with weightings ...
            #performance_per_combiner[i] = accuracy(ta_valid, comb_res)
            performance_per_combiner[i] = self.eval_metric(ta_valid, comb_res)
        self.selected_combiner = self.combiners[performance_per_combiner.argmax()]
        # Clear temporarily obtained fusion results.
        self.multi_combiner_decision_tensor = []
        # Here, the AutoCombiner could be trained on the whole dataset again.
        # super().train(decision_tensor, true_assignments)

    def combine(self, decision_tensor):
        """
        Combine decision outputs using the AutoCombiner (AC) model. Both continuous and crisp classification outputs are
        supported. This procedure involves selecting the best method regarding its classification performance in case
        of a trained AC.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: A matrix (`numpy.array`) of crisp or continuous class assignments which represents fused decisions.
                Axis 0 represents samples and axis 1 the class labels which are aligned with axis 2 in
                ``decision_tensor`` input tensor.
        """
        #super().combine(decision_tensor)  # Left for insights into performances of preselected combiners.

        if self.selected_combiner is not None:
            return self.selected_combiner.combine(decision_tensor)
        else:
            raise TypeError("No selection performed. Use train() before combining to obtain an automatic selection.")

    def set_validation_size(self, validation_size):
        """
        Set the validation size, based on which the training data is split and the best combiner is selected.

        :param validation_size: A `float` between `0` and `1.0`. Ratio of the validation data set.
        """
        self.validation_size = validation_size

    def get_selected_combiner(self):
        """
        :return: The method selected by the `AutoCombiner`.
        """
        return self.selected_combiner

    def get_eval_metric(self):
        """
        :return: The metric used for the selection of the best performing combiner.
        """
        return self.eval_metric
