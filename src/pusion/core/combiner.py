from abc import abstractmethod

from pusion.util.exceptions import ValidationError


class Combiner:
    """
    Combiner's root class. This class as well as the subclasses in this module set the structure for each combiner
    provided by the framework. It also accommodates methods and attributes which are essential for combiner's
    registration and obtainment.

    Each combiner to be registered in the framework, needs to base at least one of the following classes:

    - :class:`UtilityBasedCombiner`
    - :class:`TrainableCombiner`
    - :class:`EvidenceBasedCombiner`.

    Furthermore, it needs a `list` :attr:`_SUPPORTED_PAC` of supported PAC tuples set as a class attribute.
    A PAC tuple is a `tuple` of string constants (classification problem, assignment type and coverage type). See:

    - `pusion.util.constants.Problem`
    - `pusion.util.constants.AssignmentType`
    - `pusion.util.constants.CoverageType`

    respectively. Example of a new combiner:

    .. code:: python

        class NewCombiner(TrainableCombiner):

        _SUPPORTED_PAC = [
            (Problem.MULTI_CLASS, AssignmentType.CRISP, CoverageType.REDUNDANT),
            (Problem.MULTI_CLASS, AssignmentType.CONTINUOUS, CoverageType.REDUNDANT)
        ]

        def train(self, decision_outputs, true_assignments):
            pass

        def combine(self, decision_outputs):
            pass

    .. warning::

        Note that a new combiner also needs to be inserted into the :class:`pusion.Method` class within
        `pusion.__init__.py` file.

    """

    # List of supported PAC tuples (see above). This attribute is set by subclasses, i.e. combiner implementations.
    _SUPPORTED_PAC = []
    # List of all (combiner class, PAC)-mappings established during combiner registrations.
    _ALL_PAC_MAPPINGS = []
    # List of all combiner classes recognized by the framework.
    _ALL_COMBINERS = []

    # A shortname identifying the combiner.
    SHORT_NAME = None

    def __init__(self):
        self.pac = None
        self.coverage = None

    def __init_subclass__(cls, **kwargs):
        """
        Python-hook which registers all inheriting combiner classes defining `_SUPPORTED_PAC`.
        """
        super().__init_subclass__(**kwargs)
        if '_SUPPORTED_PAC' not in cls.__dict__:
            # A combiner class cannot be registered if the attribute _SUPPORTED_PAC is unset.
            return
        for pac in cls._SUPPORTED_PAC:
            cls._ALL_PAC_MAPPINGS.append((cls, pac))
        if cls not in cls._ALL_COMBINERS:
            cls._ALL_COMBINERS.append(cls)

    @classmethod
    def obtain(cls, config):
        """
        Obtain a combiner registered by the framework.

        :param config: :class:`pusion.model.configuration.Configuration`. User-defined configuration.
        :return: A `combiner` object.
        """
        # Search for basis combiners supporting the given problem, assignment and coverage type (pac).
        if (config.method, config.get_pac()) in cls._ALL_PAC_MAPPINGS:
            return cls.__prepare_method(config.method, config.get_pac())
        # Search for specialized combiners supporting the given problem, assignment and coverage type (pac).
        for subclass_combiner in config.method.__subclasses__():
            if (subclass_combiner, config.get_pac()) in cls._ALL_PAC_MAPPINGS:
                return cls.__prepare_method(subclass_combiner, config.get_pac())
        raise ValidationError("The given configuration is not supported by '{}'.".format(config.method.__name__))

    @classmethod
    def __prepare_method(cls, method, pac):
        combiner = method()
        combiner.pac = pac
        return combiner

    @abstractmethod
    def combine(self, decision_tensor):
        """
        Abstract method. Combine decision outputs by combiner's implementation.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of decision outputs by different classifiers per sample.
        :return: `numpy.array` of shape `(n_samples, n_classes)`. A matrix of class assignments which represents fused
                decisions obtained by combiner's implementation. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        pass

    def set_coverage(self, coverage):
        """
        Set the coverage for complementary-redundant decisions.

        :param coverage: `list` of `list` elements. Each inner list contains classes as integers covered by a
                classifier, which is identified by the positional index of the respective list.
        """
        self.coverage = coverage


class UtilityBasedCombiner(Combiner):
    """
    A combiner of type :class:`UtilityBasedCombiner` fuses decisions solely based on the outputs of the ensemble
    classifiers. It does not take any further information or evidence about respective ensemble classifiers into
    account.
    """
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        """
        Abstract method. Combine decision outputs by combiner's implementation.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of decision outputs by different classifiers per sample.
        :return: `numpy.array` of shape `(n_samples, n_classes)`. A matrix of class assignments which represents fused
                decisions obtained by combiner's implementation. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        pass


class TrainableCombiner(Combiner):
    """
    A combiner of type :class:`TrainableCombiner` needs to be trained using decision outputs of the ensemble classifiers
    with true class assignments in order to combine decisions of unknown samples.
    """
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        """
        Abstract method. Combine decision outputs by combiner's implementation.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of decision outputs by different classifiers per sample.
        :return: `numpy.array` of shape `(n_samples, n_classes)`. A matrix of class assignments which represents fused
                decisions obtained by combiner's implementation. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        pass

    @abstractmethod
    def train(self, decision_tensor, true_assignments, **kwargs):
        """
        Abstract method. Train combiner's implementation using decision outputs an appropriate true assignments.
         
        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of decision outputs by different classifiers per sample.
        :param true_assignments: `numpy.array` of shape `(n_samples, n_classes)`.
                Matrix of class assignments which are considered true for each sample during the training procedure.
        :param \*\*kwargs: The `\*\*kwargs` parameter may be used to use additional test data for the AutoFusion selection
                procedure.
        """
        pass


class EvidenceBasedCombiner(Combiner):
    """
    A combiner of type :class:`EvidenceBasedCombiner` takes an additional evidence into account while combining outputs
    of ensemble classifiers. Thus, it is able to empower better classifiers in order to obtain a fusion result with
    higher overall classification performance.
    """
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        """
        Abstract method. Combine decision outputs by combiner's implementation.

        :param decision_tensor: `numpy.array` of shape `(n_classifiers, n_samples, n_classes)`.
                Tensor of decision outputs by different classifiers per sample.
        :return: `numpy.array` of shape `(n_samples, n_classes)`. A matrix of class assignments which represents fused
                decisions obtained by combiner's implementation. Axis 0 represents samples and axis 1 the class
                assignments which are aligned with axis 2 in ``decision_tensor`` input tensor.
        """
        pass

    @abstractmethod
    def set_evidence(self, evidence):
        """
        Abstract method. Set the evidence for evidence-based combiner implementations.
        The evidence is given by confusion matrices calculated according to Kuncheva :footcite:`kuncheva2014combining`.

        .. footbibliography::

        :param evidence: `list` of `numpy.array` elements of shape `(n_classes, n_classes)`. Confusion matrices
                for each ensemble classifier.
        """
        pass
