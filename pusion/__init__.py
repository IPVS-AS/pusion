from pusion.control.decision_processor import DecisionProcessor
from pusion.core import *
from pusion.model.configuration import Configuration
from pusion.util import *
from pusion.util.constants import *
from pusion.util.exceptions import *
from pusion.evaluation.evaluation_metrics import *

from pusion.auto.auto_combiner import AutoCombiner
from pusion.auto.generic_combiner import GenericCombiner
from pusion.core.behaviour_knowledge_space_combiner import BehaviourKnowledgeSpaceCombiner
from pusion.core.borda_count_combiner import BordaCountCombiner
from pusion.core.cosine_similarity_combiner import CosineSimilarityCombiner
from pusion.core.decision_templates_combiner import DecisionTemplatesCombiner
from pusion.core.decision_tree_combiner import DecisionTreeCombiner
from pusion.core.dempster_shafer_combiner import DempsterShaferCombiner
from pusion.core.macro_majority_vote_combiner import MacroMajorityVoteCombiner
from pusion.core.maximum_likelihood_combiner import MaximumLikelihoodCombiner
from pusion.core.micro_majority_vote_combiner import MicroMajorityVoteCombiner
from pusion.core.naive_bayes_combiner import NaiveBayesCombiner
from pusion.core.neural_network_combiner import NeuralNetworkCombiner
from pusion.core.simple_average_combiner import SimpleAverageCombiner
from pusion.core.weighted_voting_combiner import WeightedVotingCombiner


# Maintain this static attributes according to implemented methods for ease of framework usage,
# e.g., `pusion.Method.AutoCombiner`.
class Method:
    BEHAVIOUR_KNOWLEDGE_SPACE = BehaviourKnowledgeSpaceCombiner
    BORDA_COUNT = BordaCountCombiner
    COSINE_SIMILARITY = CosineSimilarityCombiner
    DECISION_TEMPLATES = DecisionTemplatesCombiner
    DECISION_TREE_COMBINER = DecisionTreeCombiner
    DEMPSTER_SHAFER = DempsterShaferCombiner
    MACRO_MAJORITY_VOTE = MacroMajorityVoteCombiner
    MAXIMUM_LIKELIHOOD = MaximumLikelihoodCombiner
    MICRO_MAJORITY_VOTE = MicroMajorityVoteCombiner
    NAIVE_BAYES = NaiveBayesCombiner
    NEURAL_NETWORK = NeuralNetworkCombiner
    SIMPLE_AVERAGE = SimpleAverageCombiner
    WEIGHTED_VOTING = WeightedVotingCombiner
    AUTO = AutoCombiner
    GENERIC = GenericCombiner


class PerformanceMetric:
    PRECISION = precision
    RECALL = recall
    ACCURACY = accuracy
    F1_SCORE = f1
    JACCARD_SCORE = jaccard
    MEAN_MULTILABEL_ACCURACY = mean_multilabel_accuracy


class DiversityMetric:
    COHENS_KAPPA_MULTICLASS = pairwise_cohens_kappa_multiclass
    COHENS_KAPPA_MULTILABEL = pairwise_cohens_kappa_multilabel
    CORRELATION = correlation
    Q_STATISTIC = q_statistic
