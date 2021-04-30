from pusion.core.combiner import *
from pusion.auto.detector import *
from pusion.util.generator import *
from pusion.evaluation.evaluation_metrics import *


class AutoCombiner(TrainableCombiner, EvidenceBasedCombiner, UtilityBasedCombiner, Generic):
    """
    `AutoCombiner` allows for automatic decision fusion using all methods provided by the framework, which are
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
        self.problem = Problem.GENERIC
        self.assignment_type = AssignmentType.GENERIC
        self.coverage_type = CoverageType.REDUNDANT

        self.combiner_type_selection = []
        self.combiners = []
        self.multi_combiner_decision_tensor = []
        self.selected_combiner = None

        self.decision_tensor_validation = None
        self.true_assignments_validation = None

        self.performance_measures = None

        self.evidence = None

    def set_coverage(self, coverage):
        """
        Set the coverage in case of complementary-redundant classification data.

        :param coverage: The coverage is described by using a nested list. Each list describes the classifier based on
                its position. Elements of those lists (integers) describe the actual class coverage of the respective
                classifier. E.g., with ``[[0,1], [0,2,3]]`` the classes 0,1 are covered by the first classifier and
                0,2,3 are covered by the second one.
        """
        self.coverage = coverage
        self.coverage_type = determine_coverage_type(coverage)

    def set_evidence(self, evidence):
        """
        Set the evidence for evidence based combiners. This method preselects all combiners of type
        `EvidenceBasedCombiner`.

        :param evidence: `numpy.array` of shape `(n_classifiers, n_classes, n_classes)`.
                Confusion matrices for each of `n` classifiers.
        """
        self.__add_combiner_type(EvidenceBasedCombiner)
        self.evidence = evidence

    def train(self, decision_tensor, true_assignments):
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
        self.__add_combiner_type(UtilityBasedCombiner)
        self.__add_combiner_type(TrainableCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)
        self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)
        self.__prepare()

        # split the training data set into train and validation in order to establish evaluation
        dt_train, ta_train, self.decision_tensor_validation, self.true_assignments_validation = \
            split_into_train_and_validation_data(decision_tensor, true_assignments)

        for combiner in self.combiners:
            if isinstance(combiner, TrainableCombiner):
                combiner.train(dt_train, ta_train)

        # Evaluation phase
        if self.__combiner_type_selected(TrainableCombiner):
            self.performance_measures = np.zeros(len(self.combiners))
            for i, combiner in enumerate(self.combiners):
                res = combiner.combine(self.decision_tensor_validation)
                self.performance_measures[i] = accuracy(self.true_assignments_validation, res)

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
        self.__add_combiner_type(UtilityBasedCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)

        if not self.__combiner_type_selected(TrainableCombiner):
            self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)

        self.__prepare()

        # Combine phase
        for combiner in self.combiners:
            self.multi_combiner_decision_tensor.append(combiner.combine(decision_tensor))

        if self.__combiner_type_selected(TrainableCombiner):
            self.selected_combiner = self.combiners[self.performance_measures.argmax()]
            return self.multi_combiner_decision_tensor[self.performance_measures.argmax()]
        else:
            # As the performance is unknown, retrieve the result from a random combiner.  # TODO use heuristics...
            combiner_index = np.random.choice(len(self.multi_combiner_decision_tensor))
            self.selected_combiner = self.combiners[combiner_index]
            return self.multi_combiner_decision_tensor[combiner_index]

    def __prepare(self):
        for combiner in self.combiners:
            if isinstance(combiner, EvidenceBasedCombiner) and self.evidence:
                combiner.set_evidence(self.evidence)
            if self.coverage_type == CoverageType.COMPLEMENTARY \
                    or self.coverage_type == CoverageType.COMPLEMENTARY_REDUNDANT:
                combiner.set_coverage(self.coverage)

    def __obtain_from_registered_methods(self, pac, combiner_types):
        methods = []
        for combiner_type in combiner_types:
            for map_entry in super()._ALL_PAC_MAPPINGS:
                method, pac_ = map_entry[0], map_entry[1]
                if pac_ == pac and method not in methods and not issubclass(method, Generic) \
                        and issubclass(method, combiner_type):
                    methods.append(method)
        return [method() for method in methods]

    def __add_combiner_type(self, combiner_type):
        if combiner_type not in self.combiner_type_selection:
            self.combiner_type_selection.append(combiner_type)

    def __combiner_type_selected(self, combiner_type):
        return combiner_type in self.combiner_type_selection

    def get_pac(self):
        """
        :return: `tuple` of detected problem, assignment type and coverage type.
        """
        return self.problem, self.assignment_type, self.coverage_type

    def get_combiners(self):
        """
        :return: list of core methods preselected by the `AutoCombiner`.
        """
        return self.combiners

    def get_combiner_type_selection(self):
        """
        :return: list of combiner types established by usage.
        """
        return self.combiner_type_selection

    def get_multi_combiner_decision_tensor(self):
        """
        :return: list of `numpy.array` elements of shape `(n_samples, n_classifiers)`.
                Fusion results obtained by selected fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        return self.multi_combiner_decision_tensor

    def get_selected_combiner(self):
        """
        :return: The method selected by the `AutoCombiner`.
        """
        return self.selected_combiner
