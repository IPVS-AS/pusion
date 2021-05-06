import uuid
import warnings

import numpy as np

import pusion as p

import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *

warnings.filterwarnings('error')  # TODO delete

eval_id = uuid.uuid4().hex
n_runs = 5

np.random.seed(1)

combiners_per_run = []

classifiers_performance_run_tuples = []
classifiers_mean_confidence_run_tuples = []
combiners_performance_run_tuples = []
combiners_mean_confidence_run_tuples = []
performance_improvements = []
classifier_max_scores = []
classifier_max_mean_confidences = []
combiners_max_scores = []

ensemble_diversity_correlation_scores = []
ensemble_diversity_q_statistic_scores = []
ensemble_diversity_kappa_statistic = []
ensemble_diversity_disagreement = []
ensemble_diversity_double_fault = []

ensemble_diversity_cohens_kappa_scores = []
ensemble_pairwise_euclidean_distance = []

combiners_runtime_run_matrices = []

# plot properties
meanprops = dict(markerfacecolor='black', markeredgecolor='white')

for i in range(n_runs):
    print(">>> ", i)
    classifiers = [
        # KNeighborsClassifier(1),
        # KNeighborsClassifier(3),
        # KNeighborsClassifier(5),
        # KNeighborsClassifier(7),
        # KNeighborsClassifier(9),
        # DecisionTreeClassifier(max_depth=5),  # MLK
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # MLK
        # MLPClassifier(max_iter=5000, hidden_layer_sizes=(10,)),  # MLK
        # MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10)),  # MLK
        # MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10, 10)),  # MLK
        # LinearDiscriminantAnalysis(),
        # LogisticRegression(),
        # SVC(),
        # SVC(kernel="linear"),
        # SVC(kernel="linear"),
        # SVC(kernel="poly"),
        # SVC(kernel="rbf"),
        # SVC(kernel="sigmoid"),
        # SVC(gamma=2, C=1),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=3),
        # DecisionTreeClassifier(max_depth=2),
        # DecisionTreeClassifier(max_depth=3),
        # DecisionTreeClassifier(max_depth=4),
        # DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=1, n_estimators=10, max_features=1),
        RandomForestClassifier(max_depth=3, n_estimators=9, max_features=1),
        RandomForestClassifier(max_depth=5, n_estimators=8, max_features=1),
        RandomForestClassifier(max_depth=7, n_estimators=7, max_features=1),
        RandomForestClassifier(max_depth=10, n_estimators=6, max_features=1),
        # AdaBoostClassifier(),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        # AdaBoostClassifier(n_estimators=10),
        # AdaBoostClassifier(n_estimators=10),
        # AdaBoostClassifier(n_estimators=10),
        # AdaBoostClassifier(n_estimators=20),
        # AdaBoostClassifier(n_estimators=30),
        # AdaBoostClassifier(n_estimators=40),
        # AdaBoostClassifier(n_estimators=50),
    ]

    coverage = p.generate_classification_coverage(n_classifiers=len(classifiers),
                                                  n_classes=5,
                                                  overlap=1,
                                                  normal_class=True)

    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multiclass_ensemble_classification_outputs(
        classifiers, n_classes=5, n_samples=5000, coverage=None)

    perf_metrics = (p.PerformanceMetric.ACCURACY, p.PerformanceMetric.F1_SCORE, p.PerformanceMetric.MEAN_CONFIDENCE)

    print("============== Ensemble ================")
    eval_classifiers = Evaluation()
    eval_classifiers.set_metrics(*perf_metrics)
    eval_classifiers.set_instances(classifiers)

    eval_classifiers.evaluate(y_test, y_ensemble_test)
    print(eval_classifiers.get_report())

    print("=========== GenericCombiner ============")
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    # dp.set_parallel(False)

    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    combiners_per_run.append(dp.get_combiners())

    eval_combiner = Evaluation()
    eval_combiner.set_metrics(*perf_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, y_comb)
    print(eval_combiner.get_report())
    print("----------------------------------------")
    eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
    print(eval_combiner.get_runtime_report())
    print("========================================")

    classifiers_performance_tuples = eval_classifiers.get_top_n_instances()
    classifiers_performance_run_tuples.append(classifiers_performance_tuples)

    classifiers_mean_confidence_tuples = eval_classifiers.get_top_n_instances(
        metric=p.PerformanceMetric.MEAN_CONFIDENCE)
    classifiers_mean_confidence_run_tuples.append(classifiers_mean_confidence_tuples)

    combiners_performance_tuples = eval_combiner.get_top_n_instances()
    combiners_performance_run_tuples.append(combiners_performance_tuples)

    combiners_mean_confidence_tuples = eval_combiner.get_top_n_instances(metric=p.PerformanceMetric.MEAN_CONFIDENCE)
    combiners_mean_confidence_run_tuples.append(combiners_mean_confidence_tuples)

    classifier_max_score = classifiers_performance_tuples[0][1]
    classifier_max_scores.append(classifier_max_score)

    classifier_max_mean_confidence = classifiers_mean_confidence_tuples[0][1]
    classifier_max_mean_confidences.append(classifier_max_mean_confidence)

    combiners_max_score = combiners_performance_tuples[0][1]
    combiners_max_scores.append(combiners_max_score)

    performance_improvement = combiners_max_score - classifier_max_score
    performance_improvements.append(performance_improvement)

    ensemble_diversity_correlation_scores.append(pairwise_correlation(y_ensemble_test, y_test))
    ensemble_diversity_q_statistic_scores.append(pairwise_q_statistic(y_ensemble_test, y_test))
    ensemble_diversity_kappa_statistic.append(pairwise_kappa_statistic(y_ensemble_test, y_test))
    ensemble_diversity_disagreement.append(pairwise_disagreement(y_ensemble_test, y_test))
    ensemble_diversity_double_fault.append(pairwise_double_fault(y_ensemble_test, y_test))

    ensemble_diversity_cohens_kappa_scores.append(pairwise_cohens_kappa(y_ensemble_test))
    ensemble_pairwise_euclidean_distance.append(pairwise_euclidean_distance(y_ensemble_test))

    combiners_runtime_matrix = eval_combiner.get_runtime_matrix()
    combiners_runtime_run_matrices.append(combiners_runtime_matrix)


# === Fusion methods comparison ========================================================================================

reduced_combiners_performances = {}
for run_tuple in combiners_performance_run_tuples:  # reduce
    for t in run_tuple:
        comb_index = type(t[0])
        if comb_index not in reduced_combiners_performances:  # create a score list if non-existent
            reduced_combiners_performances[comb_index] = []
        reduced_combiners_performances[comb_index].append(t[1])

combiners = [comb for comb in reduced_combiners_performances.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performances = [reduced_combiners_performances[c] for c in combiners]

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops)
plt.title("Fusion methods comparison (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "000_box_plot_combiner_comparison", eval_id)
plt.close()

# --- Fusion methods comparison (with control) -------------------------------------------------------------------------
combiners_performances.append(classifier_max_scores)
combiners_names.append('Control')

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops)
plt.title("Framework control comparison (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "010_box_plot_combiner_control_comparison", eval_id)
plt.close()

# --- Fusion methods mean confidence comparison ------------------------------------------------------------------------
reduced_combiners_mean_confidences = {}
for run_tuple in combiners_mean_confidence_run_tuples:  # reduce
    for t in run_tuple:
        comb_index = type(t[0])
        if comb_index not in reduced_combiners_mean_confidences:  # create a score list if non-existent
            reduced_combiners_mean_confidences[comb_index] = []
        reduced_combiners_mean_confidences[comb_index].append(t[1])

combiners = [comb for comb in reduced_combiners_mean_confidences.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performances = [reduced_combiners_mean_confidences[c] for c in combiners]

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops)
plt.title("Fusion methods comparison (" + str(n_runs) + " runs)")
plt.ylabel("Mean confidence", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "020_box_plot_combiner_comparison_mean_confidence", eval_id)
plt.close()

# --- Fusion methods mean confidence comparison (with control) ---------------------------------------------------------
combiners_performances.append(classifier_max_mean_confidences)
combiners_names.append('Control')

plt.figure(figsize=(10, 4.8))
plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops)
plt.title("Framework control comparison (" + str(n_runs) + " runs)")
plt.ylabel("Mean confidence", labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
plt.tight_layout()
save(plt, "021_box_plot_combiner_control_comparison_mean_confidence", eval_id)
plt.close()


# === Performance comparison (Ensemble/Framework) ======================================================================

plt.boxplot([classifier_max_scores, combiners_max_scores], showmeans=True, meanprops=meanprops)
plt.title("Performance comparison (" + str(n_runs) + " runs)")
plt.ylabel("Max. Accuracy", labelpad=15)
plt.xticks([1, 2], ['Ensemble', 'Framework'])
plt.tight_layout()
save(plt, "030_box_plot_max_performance_comparison", eval_id)
plt.close()

# --- Performance improvement by Framework -----------------------------------------------------------------------------
plt.boxplot(performance_improvements, showmeans=True, meanprops=meanprops)
plt.title("Performance improvement (" + str(n_runs) + " runs)")
plt.ylabel("Accuracy (difference)", labelpad=15)
plt.xticks([1], ['Framework'])
plt.tight_layout()
save(plt, "031_box_plot_performance_improvement", eval_id)
plt.close()

# === Diversity -- Framework Performance ===============================================================================

plt.plot(ensemble_diversity_kappa_statistic, combiners_max_scores, 'g^')
plt.xlabel("Diversity (Kappa-statistic)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_00_div_cohens_kappa2__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_correlation_scores, combiners_max_scores, 'bs')
plt.xlabel("Diversity (Correlation)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_01_div_correlation__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_q_statistic_scores, combiners_max_scores, 'g^')
plt.xlabel("Diversity (Q-statistic)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_02_div_q_stat__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_disagreement, combiners_max_scores, 'mv')
plt.xlabel("Diversity (Disagreement)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_03_div_disagreement__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_double_fault, combiners_max_scores, 'rH')
plt.xlabel("Diversity (Double Fault)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_04_div_double_fault__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_diversity_cohens_kappa_scores, combiners_max_scores, 'ro')
plt.xlabel("Diversity (Cohen's Kappa)", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_05_div_cohens_kappa__framework_performance", eval_id)
plt.close()

plt.plot(ensemble_pairwise_euclidean_distance, combiners_max_scores, 'gD')
plt.xlabel("Mean pairwise Euclidean distance", labelpad=15)
plt.ylabel("Framework Performance (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "100_data_plot_06_euclidean_distance__framework_performance", eval_id)
plt.close()


# === Diversity -- Performance Improvement =============================================================================

plt.plot(ensemble_diversity_kappa_statistic, performance_improvements, 'ro')
plt.xlabel("Diversity (Kappa statistic)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_10_div_cohens_kappa2__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_diversity_correlation_scores, performance_improvements, 'bs')
plt.xlabel("Diversity (Correlation)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_11_div_correlation__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_diversity_q_statistic_scores, performance_improvements, 'g^')
plt.xlabel("Diversity (Q-statistic)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_12_div_q_stat__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_diversity_disagreement, performance_improvements, 'mv')
plt.xlabel("Diversity (Disagreement)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_13_div_disagreement__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_diversity_double_fault, performance_improvements, 'rH')
plt.xlabel("Diversity (Double Fault)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_14_div_double_fault__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_diversity_cohens_kappa_scores, performance_improvements, 'ro')
plt.xlabel("Diversity (Cohen's Kappa)", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_15_div_cohens_kappa__perf_improvement", eval_id)
plt.close()

plt.plot(ensemble_pairwise_euclidean_distance, performance_improvements, 'bD')
plt.xlabel("Mean pairwise Euclidean distance", labelpad=15)
plt.ylabel("Performance Improvement (Accuracy)", labelpad=15)
plt.tight_layout()
save(plt, "200_data_plot_16_euclidean_distance__perf_improvement", eval_id)
plt.close()


# === Combiner runtimes ================================================================================================

runtime_tensor = np.zeros((len(combiners_runtime_run_matrices), len(combiners_runtime_run_matrices[0]), 2))

for i, runtime_matrix in enumerate(combiners_runtime_run_matrices):
    runtime_tensor[i] = runtime_matrix

runtime_tensor = np.nan_to_num(runtime_tensor)

mean_runtime_matrix = np.nanmean(runtime_tensor, axis=0)
combiners_train_mean_runtimes = mean_runtime_matrix[:, 0]
combiners_combine_mean_runtimes = mean_runtime_matrix[:, 1]
combiners_names = [c.SHORT_NAME for c in combiners_per_run[0]]

# --- Stacked mean train and combine runtimes --------------------------------------------------------------------------
plt.figure(figsize=(10, 4.8))
plt.bar(combiners_names, combiners_train_mean_runtimes, color='#93c6ed', label="Training")
plt.bar(combiners_names, combiners_combine_mean_runtimes, color='#006aba', bottom=combiners_train_mean_runtimes,
        label="Fusion")
plt.title("Mittlere Laufzeit der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Laufzeit (s)", labelpad=15)
plt.legend(loc="upper left")
plt.tight_layout()
save(plt, "z90_stacked_bars_runtime_comparison", eval_id)
plt.close()

# --- Mean train runtimes ----------------------------------------------------------------------------------------------
non_zero_indexes = np.nonzero(combiners_train_mean_runtimes)[0]
combiners_train_mean_non_zero_runtimes = combiners_train_mean_runtimes[non_zero_indexes]
combiners_non_zero_names = [combiners_names[i] for i in non_zero_indexes]

# remove outliers
max_index = np.argmax(combiners_train_mean_non_zero_runtimes)
combiners_train_mean_non_zero_runtimes = np.delete(combiners_train_mean_non_zero_runtimes, max_index)
del combiners_non_zero_names[max_index]

plt.figure(figsize=(10, 4.8))
plt.bar(combiners_non_zero_names, combiners_train_mean_non_zero_runtimes, color='#93c6ed')
plt.title("Mittlere Trainingslaufzeit der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Laufzeit (s)", labelpad=15)
plt.tight_layout()
save(plt, "z91z_train_runtime_comparison", eval_id)
plt.close()

# --- Mean combine runtimes --------------------------------------------------------------------------------------------
plt.figure(figsize=(10, 4.8))
plt.bar(combiners_names, combiners_combine_mean_runtimes, color='#006aba')
plt.title("Mittlere Fusionslaufzeit der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Laufzeit (s)", labelpad=15)
plt.tight_layout()
save(plt, "z92_combine_runtime_comparison", eval_id)
plt.close()


print("Evaluation", eval_id, "done.")
