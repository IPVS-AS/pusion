from pusion.auto.auto_combiner import AutoCombiner
from pusion.auto.generic_combiner import GenericCombiner
from pusion.core.combiner import Combiner, EvidenceBasedCombiner, TrainableCombiner
from pusion.model.configuration import Configuration


class DecisionProcessor:
    """
    :class:`DecisionProcessor` is the main user interface of the decision fusion framework. It provides all methods
    for selecting combiners including the :ref:`AutoCombiner <ac-cref>` and the :ref:`GenericCombiner <gc-cref>`.
    It also ensures uniformity and correct use of all `pusion.core` combiners.

    :param config: :class:`pusion.model.configuration.Configuration`. User-defined configuration.
    """
    def __init__(self, config: Configuration):
        self.config = config
        self.combiner = Combiner.obtain(config)
        self.evaluation = None

    def set_coverage(self, coverage):
        """
        Set the coverage in case of complementary-redundant classification data.

        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list. E.g., with
                ``[[0,1], [0,2,3]]`` the classes 0,1 are covered by the first classifier and 0,2,3 are covered by the
                second one.
        """
        self.combiner.set_coverage(coverage)

    def set_evidence(self, evidence):
        """
        Set the evidence for evidence-based combiners. The evidence is given by confusion matrices calculated
        according to Kuncheva :footcite:`kuncheva2014combining`.

        .. footbibliography::

        :param evidence: `list` of `numpy.array` elements of shape `(n_classes, n_classes)`. Confusion matrices
                for each ensemble classifier.
        """
        if isinstance(self.combiner, EvidenceBasedCombiner):
            self.combiner.set_evidence(evidence)

    def set_data_split_ratio(self, validation_size):
        """
        Set the size of the validation data used by the AutoCombiner to evaluate all applicable fusion methods in order
        to select the combiner with the best classification performance.
        Accordingly, the other data of size `1-validation_size` is used to train all individual combiners.

        :param validation_size: A `float` between `0` and `1.0`. Ratio of the validation data set.
        """
        if isinstance(self.combiner, AutoCombiner):
            self.combiner.set_validation_size(validation_size)
        raise TypeError("No AutoCombiner configuration.")

    def train(self, y_ensemble_valid, y_valid, **kwargs):
        """
        Train the combiner model determined by the configuration.

        .. warning::

            A trainable combiner is always trained with the validation dataset provided by ensemble classifiers.

        :param y_ensemble_valid: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param y_valid: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.

        :param \*\*kwargs: The `\*\*kwargs` parameter may be used to use additional test data for the AutoFusion selection
                procedure.
        """
        if isinstance(self.combiner, TrainableCombiner):
            if isinstance(self.combiner, AutoCombiner):
                self.combiner.train(y_ensemble_valid, y_valid, **kwargs)
            else:
                self.combiner.train(y_ensemble_valid, y_valid)

    def combine(self, y_ensemble_test):
        """
        Combine decision outputs using the combiner model determined by the configuration.

        :param y_ensemble_test: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: `numpy.array` of shape `(n_samples, n_classes)`. A matrix of crisp or continuous class assignments
                which represents fused decisions. Axis 0 represents samples and axis 1 the class labels which are
                aligned with axis 2 in ``y_ensemble_test`` input tensor.
        """
        return self.combiner.combine(y_ensemble_test)

    def get_multi_combiner_decision_output(self):
        """
        Retrieve the decision fusion outputs obtained by multiple combiners. This function is only callable for
        configurations including `AutoCombiner` or `GenericCombiner` as a method.

        :return: `list` of `numpy.array` elements of shape `(n_samples, n_classes)`.
                Fusion results obtained by multiple fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_multi_combiner_decision_tensor()
        raise TypeError("No multi combiner output. Use combine(...) to retrieve the output of a single combiner.")

    def get_optimal_combiner(self, eval_metric=None):
        """
        Retrieve the combiner with the best classification performance obtained by the framework, i.e. the
        `AutoCombiner` or the `GenericCombiner`.
        In case of combining with the `GenericCombiner`, an `Evaluation` needs to be set by ``set_evaluation``.

        :return: The combiner object.
        """
        if type(self.combiner) is GenericCombiner and self.evaluation is None:
            raise TypeError("No evaluation set for determining the optimal method using GenericCombiner.")
        if type(self.combiner) is GenericCombiner:
            return self.evaluation.get_top_n_instances(n=1, metric=eval_metric)[0][0]
        elif type(self.combiner) is AutoCombiner:
            return self.combiner.get_selected_combiner()
        else:
            return self.combiner

    def get_combiners(self):
        """
        Retrieve combiners (core methods) which are preselected by the framework according to the auto-detected
        configuration.
        :return: `list` of combiner objects obtained by the `GenericCombiner` or `AutoCombiner`.
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_combiners()
        raise TypeError("get_combiners() is not callable for a core combiner.")

    def get_combiner(self):
        """
        :return: Selected combiner object.
        """
        return self.combiner


    # def get_evaluation_results(self):
    #     """
    #     getter method to return the evaluation results of the combiners contained in the decision processor.
    #     :return: dict containing the evaluation reulsts of the combiners
    #     """
    #     if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
    #         problem, assignment_type, coverage_type = self.combiner.get_pac()
    #         combiner_type_selection = self.combiner.get_combiner_type_selection()
    #         optimal_comb = self.get_optimal_combiner()
    #
    #         report_dict = {
    #             'Problem': problem,
    #             'Assignment type': assignment_type,
    #             'Coverage type': coverage_type,
    #             'Combiner type selection': ', '.join([ct.__name__ for ct in combiner_type_selection]),
    #             'Compatible combiners': ', '.join([type(comb).__name__ for comb in self.get_combiners()]),
    #             'Optimal combiner': type(optimal_comb).__name__,
    #             'Classification performance': self.evaluation.get_performance_matrix(),
    #             'Evaluation metrics': self.evaluation.get_metrics(),
    #             'Instances': self.evaluation.get_instances()
    #         }
    #         return report_dict
    #     raise TypeError("get_evaluation_results() is not callable.")



    def report(self, eval_metric=None):
        """
        :return: The textual evaluation report.
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            problem, assignment_type, coverage_type = self.combiner.get_pac()
            combiner_type_selection = self.combiner.get_combiner_type_selection()
            optimal_comb = self.get_optimal_combiner(eval_metric=eval_metric)
            selection_metric = self.combiner.get_eval_metric().__name__ if isinstance(self.combiner, AutoCombiner) else (eval_metric.__name__ if eval_metric is not None else self.evaluation.metrics[0].__name__)

            report_dict = {
                'Problem': problem,
                'Assignment type': assignment_type,
                'Coverage type': coverage_type,
                'Combiner type selection': ', '.join([ct.__name__ for ct in combiner_type_selection]),
                'Compatible combiners': ', '.join([type(comb).__name__ for comb in self.get_combiners()]),
                'Selection metric': selection_metric,
                'Optimal combiner': type(optimal_comb).__name__,
                'Classification performance': '\n' + str(self.evaluation.get_report())
            }

            report_str = " " + type(self.combiner).__name__ + " - Report "
            report_str = report_str.center(90, '=') + "\n"
            tab_len = max([len(s) for s in report_dict.keys()])
            for k, v in report_dict.items():
                format_str = "{:>" + str(tab_len) + "}: {}\n"
                report_str += format_str.format(k, v)
            report_str += '=' * 90

            return report_str
        raise TypeError("report() is not callable for a core combiner.")

    def info(self):
        """
        Retrieve the information, the automatic combiner selection is based on.

        :return: `tuple` of the form `((A, B, C), D)`, whereby `A` represents the classification problem
                `('MULTI_CLASS' or 'MULTI_LABEL')`, `B` the assignment type `('CRISP' or 'CONTINUOUS')` and `C`
                the coverage type `('REDUNDANT', 'COMPLEMENTARY' or 'COMPLEMENTARY_REDUNDANT')`.

                `D` contains the combiner type selection as a `list`.
                Possible combiner types are `UtilityBasedCombiner`, `TrainableCombiner` and `EvidenceBasedCombiner`.
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_pac(), self.combiner.get_combiner_type_selection()
        raise TypeError("info() is not callable for a core combiner.")

    def get_multi_combiner_runtimes(self):
        """
        Retrieve the train and combine runtime for each combiner used during a generic fusion.

        :return: A `tuple` of two lists of tuples describing the train and combine runtimes respectively.
                Each inner tuple key value indexes the list of preselected fusion methods
                (retrievable by ``get_combiners()``).
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_multi_combiner_runtimes()
        raise TypeError("info_runtime() is not callable for a core combiner.")

    def set_evaluation(self, evaluation):
        """
        :param evaluation: :class:`pusion.control.evaluation.Evaluation` object, a combiner evaluation was
                performed with.
        """
        self.evaluation = evaluation

    def set_parallel(self, parallel=True):
        """
        Set whether the training and the combining of selected combiners should be executed sequentially or in parallel.
        :param parallel: If `True`, training and combining is performed in parallel respectively. Otherwise in sequence.
        """
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            self.combiner.set_parallel(parallel)
        else:
            raise TypeError("set_parallel() is not callable for a core combiner.")
