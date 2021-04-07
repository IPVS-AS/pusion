from pusion.core.combiner import *
from pusion.util.functions import *
from pusion.evaluation.evaluation_metrics import *


class AutoCombiner(TrainableCombiner, EvidenceBasedCombiner, UtilityBasedCombiner, Generic):
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
        self.true_assignment_validation = None

        self.performance_measures = None

        self.evidence = None

    def set_coverage(self, coverage):
        self.coverage = coverage
        self.coverage_type = determine_coverage_type(coverage)

    def set_evidence(self, evidence):
        self.__add_combiner_type(EvidenceBasedCombiner)
        self.evidence = evidence

    def train(self, decision_tensor, true_assignment):
        self.__add_combiner_type(UtilityBasedCombiner)
        self.__add_combiner_type(TrainableCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)
        self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)
        self.__prepare()

        # split the training data set into train and validation in order to establish evaluation
        dt_train, ta_train, self.decision_tensor_validation, self.true_assignment_validation = \
            split_into_train_and_validation_data(decision_tensor, true_assignment)

        for combiner in self.combiners:
            if isinstance(combiner, TrainableCombiner):
                combiner.train(dt_train, ta_train)

    def combine(self, decision_tensor):
        self.__add_combiner_type(UtilityBasedCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)

        # Evaluation phase
        if self.__combiner_type_selected(TrainableCombiner):
            self.performance_measures = np.zeros(len(self.combiners))
            for i, combiner in enumerate(self.combiners):
                res = combiner.combine(self.decision_tensor_validation)
                self.performance_measures[i] = accuracy(self.true_assignment_validation, res)

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
        return self.problem, self.assignment_type, self.coverage_type

    def get_combiners(self):
        return self.combiners

    def get_combiner_type_selection(self):
        return self.combiner_type_selection

    def get_multi_combiner_decision_tensor(self):
        return self.multi_combiner_decision_tensor

    def get_selected_combiner(self):
        return self.selected_combiner
