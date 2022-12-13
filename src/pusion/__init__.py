from pusion.auto.auto_combiner import AutoCombiner
from pusion.auto.generic_combiner import GenericCombiner
from pusion.control.decision_processor import DecisionProcessor
from pusion.core import *
from pusion.core.behaviour_knowledge_space_combiner import BehaviourKnowledgeSpaceCombiner
from pusion.core.borda_count_combiner import BordaCountCombiner
from pusion.core.cosine_similarity_combiner import CosineSimilarityCombiner
from pusion.core.decision_templates_combiner import DecisionTemplatesCombiner
from pusion.core.k_nearest_neighbors_combiner import KNNCombiner
from pusion.core.dempster_shafer_combiner import DempsterShaferCombiner
from pusion.core.macro_majority_vote_combiner import MacroMajorityVoteCombiner
from pusion.core.maximum_likelihood_combiner import MaximumLikelihoodCombiner
from pusion.core.micro_majority_vote_combiner import MicroMajorityVoteCombiner
from pusion.core.naive_bayes_combiner import NaiveBayesCombiner
from pusion.core.neural_network_combiner import NeuralNetworkCombiner
from pusion.core.simple_average_combiner import SimpleAverageCombiner
from pusion.core.weighted_voting_combiner import WeightedVotingCombiner
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.model.configuration import Configuration
from pusion.util import *
from pusion.util.constants import *
from pusion.util.exceptions import *
from pusion.util.generator import *

# Maintain this static attributes according to implemented methods for ease of framework usage,
# e.g., `pusion.Method.AutoCombiner`.
class Method:
    BEHAVIOUR_KNOWLEDGE_SPACE = BehaviourKnowledgeSpaceCombiner
    BORDA_COUNT = BordaCountCombiner
    COSINE_SIMILARITY = CosineSimilarityCombiner
    DECISION_TEMPLATES = DecisionTemplatesCombiner
    DEMPSTER_SHAFER = DempsterShaferCombiner
    KNN = KNNCombiner
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
    MICRO_PRECISION = micro_precision
    MICRO_RECALL = micro_recall
    MICRO_F1_SCORE = micro_f1
    MICRO_F2_SCORE = micro_f2
    MICRO_JACCARD_SCORE = micro_jaccard
    MACRO_PRECISION = macro_precision
    MACRO_RECALL = macro_recall
    MACRO_F1_SCORE = macro_f1
    MACRO_F2_SCORE = macro_f2
    WEIGHTED_F1_SCORE = weighted_f1
    MACRO_JACCARD_SCORE = macro_jaccard
    WEIGTED_JACCARD_SCORE = weighted_jaccard
    HAMMING_LOSS = hamming
    ACCURACY = accuracy
    MEAN_MULTILABEL_ACCURACY = mean_multilabel_confusion_matrix
    MEAN_CONFIDENCE = mean_confidence
    BALANCED_MULTICLASS_ACCURACY_SCORE = balanced_multiclass_accuracy
    ERROR_RATE = error_rate
    MULTILABEL_BRIER_SCORE_MICRO = multi_label_brier_score_micro
    MULTILABEL_BRIER_SCORE = multi_label_brier_score
    MULTICLASS_BRIER_SCORE = multiclass_brier_score
    FALSE_ALARM_RATE = far
    MULTICLASS_FDR = multiclass_fdr
    MULTILABEL_SUBSET_FDR = multilabel_subset_fdr
    MULTILABEL_MINOR_FDR = multilabel_minor_fdr
    MULTICLASS_WEIGHTED_PRECISION = multiclass_weighted_precision
    MULTILABEL_WEIGHTED_PRECISION = multi_label_weighted_precision
    MULTICLASS_CLASS_WISE_PRECISION = multiclass_class_wise_precision
    MULTILABEL_CLASS_WISE_PRECISION = multi_label_class_wise_precision
    MULTICLASS_RECALL = multiclass_recall
    MULTILABEL_RECALL = multi_label_recall
    MULTICLASS_CLASS_WISE_RECALL = multiclass_class_wise_recall
    MULTILABEL_CLASS_WISE_RECALL = multi_label_class_wise_recall
    MULTICLASS_WEIGHTED_SCIKIT_AUC_ROC_SCORE = multiclass_weighted_scikit_auc_roc_score
    MULTILABEL_WEIGHTED_PYTORCH_AUC_ROC_SCORE = multi_label_weighted_pytorch_auc_roc_score
    MULTILABEL_PYTORCH_AUC_ROC_SCORE = multi_label_pytorch_auc_roc_score
    MULTICLASS_CLASS_WISE_AVG_PRECISION = multiclass_class_wise_avg_precision
    MULTICLASS_WEIGHTED_AVG_PRECISION = multiclass_weighted_avg_precision
    MULTICLASS_AUC_PRECISION_RECALL_CURVE = multiclass_auc_precision_recall_curve
    MULTICLASS_WEIGHTED_PYTORCH_AUC_ROC = multiclass_weighted_pytorch_auc_roc
    MULTICLASS_PYTORCH_AUC_ROC = multiclass_pytorch_auc_roc
    MULTILABEL_RANKING_AVG_PRECISION_SCORE = multi_label_ranking_avg_precision_score
    MULTILABEL_RANKING_LOSS = multi_label_ranking_loss
    MULTILABEL_NORMALIZED_DISCOUNTED_CUMULATIVE_GAIN = multi_label_normalized_discounted_cumulative_gain
    MULTICLASS_TOP_1_ACCURACY = multiclass_top_1_accuracy
    MULTICLASS_TOP_3_ACCURACY = multiclass_top_3_accuracy
    MULTICLASS_TOP_5_ACCURACY = multiclass_top_5_accuracy
    MULTICLASS_LOG_LOSS = multiclass_log_loss
    MULTILABEL_LOG_LOSS = multi_label_log_loss



class DiversityMetric:
    PAIRWISE_COHENS_KAPPA = pairwise_cohens_kappa
    PAIRWISE_CORRELATION = pairwise_correlation
    PAIRWISE_Q_STATISTIC = pairwise_q_statistic
    PAIRWISE_KAPPA_STATISTIC = pairwise_kappa_statistic
    PAIRWISE_DISAGREEMENT = pairwise_disagreement
    PAIRWISE_DOUBLE_FAULT = pairwise_double_fault
    PAIRWISE_EUCLIDEAN_DISTANCE = pairwise_euclidean_distance
