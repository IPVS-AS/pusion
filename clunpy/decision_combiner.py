from clunpy.constants import *
from clunpy.core.behaviour_knowledge_space_fusion import *
from clunpy.core.borda_count_fusion import BordaCountCombiner
from clunpy.core.cosine_similarity_fusion import CosineSimilarityCombiner
from clunpy.core.decision_templates_fusion import DecisionTemplatesCombiner
from clunpy.core.dempster_shafer_fusion import DempsterShaferCombiner
from clunpy.core.macro_majority_vote_fusion import MacroMajorityVoteCombiner
from clunpy.core.micro_majority_vote_fusion import MicroMajorityVoteCombiner
from clunpy.core.naive_bayes_fusion import NaiveBayesCombiner
from clunpy.core.simple_average import SimpleAverageCombiner
from clunpy.core.weighted_voting_fusion import WeightedVotingCombiner


class DecisionCombiner:
    def __init__(self):
        pass

    def combine(self, predictions, method=Method.AUTO, problem=Problem.MULTI_CLASS, train_predictions=None,
                train_labels=None, evidence=None):
        fused_decisions = None

        if problem == Problem.MULTI_CLASS:
            if method == Method.BEHAVIOUR_KNOWLEDGE_SPACE:
                if train_predictions is not None and train_labels is not None:
                    combiner = BehaviourKnowledgeSpaceCombiner()
                    combiner.train(train_predictions, train_labels)
                    fused_decisions = combiner.combine(predictions)
                else:
                    raise RuntimeError("Learning model. Provide train_predictions and train_labels to train.")

            elif method == Method.BORDA_COUNT:
                combiner = BordaCountCombiner()
                fused_decisions = combiner.combine(predictions)

            elif method == Method.COS_SIMILARITY:
                combiner = CosineSimilarityCombiner()
                fused_decisions = combiner.combine(predictions)

            elif method == Method.DECISION_TEMPLATES:
                if train_predictions is not None and train_labels is not None:
                    combiner = DecisionTemplatesCombiner()
                    combiner.train(train_predictions, train_labels)
                    fused_decisions = combiner.combine(predictions)
                else:
                    raise RuntimeError("Learning model. Provide train_predictions and train_labels to train.")

            elif method == Method.DEMPSTER_SHAFER:
                if train_predictions is not None and train_labels is not None:
                    combiner = DempsterShaferCombiner()
                    combiner.train(train_predictions, train_labels)
                    fused_decisions = combiner.combine(predictions)
                else:
                    raise RuntimeError("Learning model. Provide train_predictions and train_labels to train.")

            elif method == Method.MACRO_MAJORITY_VOTE:
                combiner = MacroMajorityVoteCombiner()
                fused_decisions = combiner.combine(predictions)

            elif method == Method.MICRO_MAJORITY_VOTE:
                combiner = MicroMajorityVoteCombiner()
                fused_decisions = combiner.combine(predictions)

            elif method == Method.NAIVE_BAYES:
                if train_predictions is not None and train_labels is not None:
                    combiner = NaiveBayesCombiner()
                    combiner.train(train_predictions, train_labels)
                    fused_decisions = combiner.combine(predictions)
                else:
                    raise RuntimeError("Learning model. Provide train_predictions and train_labels to train.")

            elif method == Method.SIMPLE_AVERAGE:
                combiner = SimpleAverageCombiner()
                fused_decisions = combiner.combine(predictions)

            elif method == Method.WEIGHTED_VOTING:
                if evidence is not None:
                    combiner = WeightedVotingCombiner()
                    fused_decisions = combiner.combine(predictions, evidence)
                elif train_predictions is not None and train_labels is not None:
                    # TODO Implement accuracy calculation.
                    raise NotImplementedError("Evidence by train_predictions and train_labels is not supported yes."
                                              "Please provide explicit evidence for this model.")
                else:
                    raise RuntimeError("Evidence based model. Provide either an evidence vector or train_predictions "
                                       "and train_labels for this model.")

        elif problem == Problem.MULTI_LABEL:
            pass  # TODO

        return fused_decisions
