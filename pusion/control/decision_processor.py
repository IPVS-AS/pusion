from pusion.auto.auto_combiner import AutoCombiner
from pusion.auto.generic_combiner import GenericCombiner
from pusion.control.evaluation import Evaluation
from pusion.core.combiner import Combiner, EvidenceBasedCombiner, TrainableCombiner
from pusion.model.configuration import Configuration


class DecisionProcessor:
    def __init__(self, config: Configuration):
        self.config = config
        self.combiner = Combiner.obtain(config)
        self.eval = Evaluation()

    def set_coverage(self, coverage):
        self.combiner.set_coverage(coverage)

    def set_evidence(self, evidence):
        if isinstance(self.combiner, EvidenceBasedCombiner):
            self.combiner.set_evidence(evidence)

    def train(self, y_ensemble_valid, y_valid):
        if isinstance(self.combiner, TrainableCombiner):
            self.combiner.train(y_ensemble_valid, y_valid)

    def combine(self, y_ensemble_test):
        return self.combiner.combine(y_ensemble_test)

    def get_multi_combiner_decision_output(self):
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_multi_combiner_decision_tensor()
        raise TypeError("No multi combiner output. Use combine(...) to retrieve the output of a single combiner.")

    def get_optimal_combiner(self):
        if isinstance(self.combiner, GenericCombiner):
            return self.eval.get_top_n_instances(1)[0]
        elif isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_selected_combiner()
        else:
            return self.combiner

    def get_combiners(self):
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_combiners()
        raise TypeError("get_combiners() is not callable for a single combiner choice.")

    def get_combiner(self):
        return self.combiner

    def report(self):
        return self.eval.get_report()

    def info(self):
        if isinstance(self.combiner, GenericCombiner) or isinstance(self.combiner, AutoCombiner):
            return self.combiner.get_pac(), self.combiner.get_combiner_type_selection()

    def set_performance_metrics(self, *argv):
        self.eval.set_metrics(*argv)
