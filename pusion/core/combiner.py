from abc import abstractmethod

from pusion.util.exceptions import ValidationError


class Combiner:
    _SUPPORTED_PAC = []
    _ALL_PAC_MAPPINGS = []
    _ALL_COMBINERS = []

    def __init__(self):
        self.pac = None
        self.coverage = None

    def __init_subclass__(cls, **kwargs):
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
        pass

    def set_coverage(self, coverage):
        self.coverage = coverage


class UtilityBasedCombiner(Combiner):
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        pass


class TrainableCombiner(Combiner):
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        pass

    @abstractmethod
    def train(self, decision_tensor, true_assignment):
        pass


class EvidenceBasedCombiner(Combiner):
    _SUPPORTED_PAC = []

    def __init__(self):
        super().__init__()

    @abstractmethod
    def combine(self, decision_tensor):
        pass

    @abstractmethod
    def set_evidence(self, evidence):
        pass


class ComplementaryRedundantCombiner:
    @abstractmethod
    def set_coverage(self, coverage):
        pass


class Generic:
    pass
