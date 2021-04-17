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

    def train(self, y_ensemble_valid, y_valid):
        """
        Train the combiner model determined by the configuration.

        .. warning::

            A trainable combiner is always trained with the validation dataset provided by ensemble classifiers.

        :param y_ensemble_valid: `numpy.array` of shape `(n_classifier, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param y_valid: `numpy.array` of shape `(n_classifier, n_samples)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        if isinstance(self.combiner, TrainableCombiner):
            self.combiner.train(y_ensemble_valid, y_valid)

    def combine(self, y_ensemble_test):
        """
        Combine decision outputs using the combiner model determined by the configuration.

        :param y_ensemble_test: `numpy.array` of shape `(n_classifier, n_samples, n_classes)` or a `list` of
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

    def get_optimal_combiner(self):
        """
        Retrieve the combiner with the best classification performance obtained by the framework, i.e. the
        `AutoCombiner` or the `GenericCombiner`.
        In case of combining with the `GenericCombiner`, an `Evaluation` needs to be set by ``set_evaluation``.

        :return: The combiner object.
        """
        if isinstance(self.combiner, GenericCombiner) and self.evaluation is None:
            raise TypeError("No evaluation set for determining the optimal method using GenericCombiner.")
        if isinstance(self.combiner, GenericCombiner):
            return self.evaluation.get_top_n_instances(1)[0]
        elif isinstance(self.combiner, AutoCombiner):
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

    def report(self):
        """
        :return: The textual evaluation report.
        """
        return self.evaluation.get_report()

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

    def set_evaluation(self, evaluation):
        """
        :param evaluation: :class:`pusion.control.evaluation.Evaluation` object, a combiner evaluation was
                performed with.
        """
        self.evaluation = evaluation
