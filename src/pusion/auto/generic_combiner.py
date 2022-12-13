import multiprocessing as mp

from pusion.auto.detector import *
from pusion.core.combiner import *
from pusion.util.processes import *


class GenericCombiner(TrainableCombiner, EvidenceBasedCombiner, UtilityBasedCombiner):
    """
    The `GenericCombiner` (GC) allows for automatic decision fusion using all methods provided by the framework, which
    are applicable to the given problem. The key feature of this combiner is the transparency in terms of it's outer
    behaviour. Based on the usage (i.e. method calls) and the automatically detected configuration,
    the `GenericCombiner` preselects all compatible methods from `pusion.core`. The main purpose is to retrieve fusion
    results obtained by the all applicable methods. The main difference to the `AutoCombiner` is that decision fusion
    results are handed over to the user for further comparison and selection. Thus, GC is not suitable for
    the online fusion.
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

        self.multi_combiner_train_runtimes = []
        self.multi_combiner_combine_runtimes = []

        self.evidence = None
        self.parallel = True

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
        Train the Generic Combiner. This method detects the configuration based on the ``decision_tensor`` and
        trains all trainable combiners that are applicable to this configuration.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of either crisp or continuous class assignments which are considered true for each sample during
                the training procedure.
        """
        if self.parallel:
            self.train_par(decision_tensor, true_assignments)
        else:
            self.train_seq(decision_tensor, true_assignments)

    def train_par(self, decision_tensor, true_assignments):
        """
        Train the Generic Combiner by training individual combiners in parallel.
        This method detects the configuration based on the ``decision_tensor`` and trains all trainable combiners
        that are applicable to this configuration.

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

        # Create a thread-safe queue for combiners
        queue = mp.Manager().Queue()
        processes = []
        for i, combiner in enumerate(self.combiners):
            if isinstance(combiner, TrainableCombiner):
                process = mp.Process(target=p_train, args=(i, combiner, decision_tensor, true_assignments, queue))
                processes.append(process)
            else:
                queue.put((i, combiner, np.nan))
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.combiners = [None for _ in range(len(self.combiners))]
        while not queue.empty():
            i, combiner, t_elapsed = queue.get()
            self.combiners[i] = combiner
            self.multi_combiner_train_runtimes.append((i, t_elapsed))

    def train_seq(self, decision_tensor, true_assignments):
        """
        Train the Generic Combiner by training individual combiners in sequence.
        This method detects the configuration based on the ``decision_tensor`` and trains all trainable combiners
        that are applicable to this configuration.

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

        self.multi_combiner_train_runtimes = []
        for i, combiner in enumerate(self.combiners):
            if isinstance(combiner, TrainableCombiner):
                t_begin = time.perf_counter()
                combiner.train(decision_tensor, true_assignments)
                t_elapsed = time.perf_counter() - t_begin
                self.multi_combiner_train_runtimes.append((i, t_elapsed))

    def combine(self, decision_tensor):
        """
        Combine decision outputs using the AutoCombiner (AC) model. Both continuous and crisp classification outputs are
        supported. This procedure involves combining decision outputs by each individual method which is applicable
        to the detected configuration.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: list of `numpy.array` of shape `(n_samples, n_classifiers)`.
                Fusion results obtained by selected fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        if self.parallel:
            return self.combine_par(decision_tensor)
        else:
            return self.combine_seq(decision_tensor)

    def combine_par(self, decision_tensor):
        """
        Combine decision outputs by GC. Both continuous and crisp classification outputs are supported.
        This procedure involves combining decision outputs by each individual method which is applicable
        to the detected configuration. Each combine procedure is spawned in a separate thread and thus performed
        in parallel.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: list of `numpy.array` of shape `(n_samples, n_classifiers)`.
                Fusion results obtained by selected fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        self.__add_combiner_type(UtilityBasedCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)

        if not self.__combiner_type_selected(TrainableCombiner):
            self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)

        self.__prepare()

        # Create a thread-safe queue for combiners
        queue = mp.Manager().Queue()
        processes = []
        for i, combiner in enumerate(self.combiners):
            process = mp.Process(target=p_combine, args=(i, combiner, decision_tensor, queue))
            processes.append(process)
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        self.multi_combiner_decision_tensor = [None for _ in range(len(self.combiners))]
        while not queue.empty():
            i, decision_matrix, t_elapsed = queue.get()
            self.multi_combiner_decision_tensor[i] = decision_matrix
            self.multi_combiner_combine_runtimes.append((i, t_elapsed))
        return self.multi_combiner_decision_tensor

    def combine_seq(self, decision_tensor):
        """
        Combine decision outputs by GC. Both continuous and crisp classification outputs are supported.
        This procedure involves combining decision outputs by each individual method which is applicable
        to the detected configuration. Each combine procedure is initiated in sequence.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)` or a `list` of
                `numpy.array` elements of shape `(n_samples, n_classes')`, where `n_classes'` is classifier-specific
                due to the coverage.

                Tensor of either crisp or continuous decision outputs by different classifiers per sample.

        :return: list of `numpy.array` of shape `(n_samples, n_classifiers)`.
                Fusion results obtained by selected fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        self.__add_combiner_type(UtilityBasedCombiner)
        self.problem = determine_problem(decision_tensor)
        self.assignment_type = determine_assignment_type(decision_tensor)

        if not self.__combiner_type_selected(TrainableCombiner):
            self.combiners = self.__obtain_from_registered_methods(self.get_pac(), self.combiner_type_selection)

        self.__prepare()

        self.multi_combiner_decision_tensor = []
        self.multi_combiner_combine_runtimes = []
        for i, combiner in enumerate(self.combiners):
            t_begin = time.perf_counter()
            self.multi_combiner_decision_tensor.append(combiner.combine(decision_tensor))
            t_elapsed = time.perf_counter() - t_begin
            self.multi_combiner_combine_runtimes.append((i, t_elapsed))
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
                if pac_ == pac and method not in methods and not issubclass(method, GenericCombiner) \
                        and method is not GenericCombiner and issubclass(method, combiner_type):
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
        :return: list of `numpy.array` of shape `(n_samples, n_classifiers)`.
                Fusion results obtained by selected fusion methods.
                The list is aligned with the list of preselected fusion methods (retrievable by ``get_combiners()``).
        """
        return self.multi_combiner_decision_tensor

    def get_multi_combiner_runtimes(self):
        """
        :return: A `tuple` of two lists of tuples describing the train and combine runtimes respectively.
                Each inner tuple key value indexes the list of preselected fusion methods
                (retrievable by ``get_combiners()``).
        """
        return self.multi_combiner_train_runtimes, self.multi_combiner_combine_runtimes

    def set_parallel(self, parallel=True):
        """
        Set whether the training and the combining of selected combiners should be executed sequentially or in parallel.
        :param parallel: If `True`, training and combining is performed in parallel respectively. Otherwise in sequence.
        """
        self.parallel = parallel
