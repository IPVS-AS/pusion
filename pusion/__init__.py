from pusion.control.decision_processor import DecisionProcessor
from pusion.evaluation.evaluation import Evaluation
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
    DECISION_TREE = DecisionTreeCombiner
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
    MEAN_MULTILABEL_ACCURACY = mean_multilabel_confusion_matrix
    MEAN_CONFIDENCE = mean_confidence


class DiversityMetric:
    PAIRWISE_COHENS_KAPPA = pairwise_cohens_kappa
    PAIRWISE_CORRELATION = pairwise_correlation
    PAIRWISE_Q_STATISTIC = pairwise_q_statistic
    PAIRWISE_KAPPA_STATISTIC = pairwise_kappa_statistic
    PAIRWISE_DISAGREEMENT = pairwise_disagreement
    PAIRWISE_DOUBLE_FAULT = pairwise_double_fault
    PAIRWISE_EUCLIDEAN_DISTANCE = pairwise_euclidean_distance
