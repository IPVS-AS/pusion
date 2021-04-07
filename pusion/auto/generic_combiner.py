from pusion.core.combiner import *
from pusion.util.functions import *


class GenericCombiner(TrainableCombiner, EvidenceBasedCombiner, UtilityBasedCombiner, Generic):
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

        for combiner in self.combiners:
            if isinstance(combiner, TrainableCombiner):
                combiner.train(decision_tensor, true_assignment)

    def combine(self, decision_tensor):
        self.__add_combiner_type(UtilityBasedCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)

        if not self.__combiner_type_selected(TrainableCombiner):
            self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)

        self.__prepare()

        # Combine phase
        for combiner in self.combiners:
            self.multi_combiner_decision_tensor.append(combiner.combine(decision_tensor))

        return self.multi_combiner_decision_tensor

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
