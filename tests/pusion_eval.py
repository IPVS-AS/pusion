import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *

warnings.filterwarnings('error')  # halt on warning

eval_id = time.strftime("%Y%m%d-%H%M%S")

n_runs = 50
n_classes = 5
n_samples = 2000
random_state = 1
cr = False
perf_metrics = (p.PerformanceMetric.ACCURACY, p.PerformanceMetric.MICRO_F1_SCORE, p.PerformanceMetric.MEAN_CONFIDENCE)

# ----------------------------------------------------------------------------------------------------------------------
eval_dict = {}

combiners_per_run = []
classifiers_performance_run_tuples = []
classifiers_mean_confidence_run_tuples = []
combiners_performance_run_tuples = []
combiners_mean_confidence_run_tuples = []
performance_improvements = []
classifier_max_scores = []
classifier_max_mean_confidences = []
combiners_max_scores = []
classifier_score_stds = []

ensemble_mean_scores = []

best_combiners_per_run = []
best_combiner_perf_tuples_per_run = []

ensemble_diversity_correlation_scores = []
ensemble_diversity_q_statistic_scores = []
ensemble_diversity_kappa_statistic = []
ensemble_diversity_disagreement = []
ensemble_diversity_double_fault = []

ensemble_diversity_cohens_kappa_scores = []
ensemble_pairwise_euclidean_distance = []

combiners_runtime_run_matrices = []


np.random.seed(random_state)

coverage_overlaps = [i/n_runs for i in range(n_runs)]
coverage_list = []
for i in range(n_runs):
    coverage_list.append(p.generate_classification_coverage(n_classifiers=5,
                                                            n_classes=n_classes,
                                                            overlap=coverage_overlaps[i],
                                                            normal_class=True))
np.random.seed(random_state)


for i in range(n_runs):
    print(">>> ", i)

    if cr:
        # hold same classifier initializations over runs for figuring out effects of different coverages.
        np.random.seed(random_state)

    classifiers = [
        # KNeighborsClassifier(3),
        # DecisionTreeClassifier(max_depth=5),  # MLK
        # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # MLK
        MLPClassifier(max_iter=5000, random_state=1, hidden_layer_sizes=(20, 20, 20)),  # MLK
        MLPClassifier(max_iter=5000, random_state=2, hidden_layer_sizes=(20, 20, 20)),  # MLK
        MLPClassifier(max_iter=5000, random_state=3, hidden_layer_sizes=(20, 20, 20)),  # MLK
        MLPClassifier(max_iter=5000, random_state=4, hidden_layer_sizes=(20, 20, 20)),  # MLK
        MLPClassifier(max_iter=5000, random_state=5, hidden_layer_sizes=(20, 20, 20)),  # MLK
        # LinearDiscriminantAnalysis(),
        # LogisticRegression(),
        # SVC(kernel="rbf"),
        # SVC(kernel="sigmoid"),
        # GaussianProcessClassifier(1.0 * RBF(1.0)),
        # DecisionTreeClassifier(max_depth=3),
        # RandomForestClassifier(max_depth=3, n_estimators=9, random_state=1),
        # GaussianNB(),
        # QuadraticDiscriminantAnalysis(),
        # AdaBoostClassifier(n_estimators=20),
    ]

    coverage = coverage_list[i]

    y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
        p.generate_multiclass_ensemble_classification_outputs(classifiers, n_classes, n_samples)
    # y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
    #     p.generate_multiclass_cr_ensemble_classification_outputs(classifiers, n_classes, n_samples, coverage)

    print("============== Ensemble ================")
    eval_classifiers = Evaluation()
    eval_classifiers.set_metrics(*perf_metrics)
    if cr:
        eval_classifiers.set_instances('Ensemble')
        eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
    else:
        eval_classifiers.set_instances(classifiers)
        eval_classifiers.evaluate(y_test, y_ensemble_test)
    print(eval_classifiers.get_report())

    print("=========== GenericCombiner ============")
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.set_parallel(True)
    if cr:
        dp.set_coverage(coverage)
    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    combiners_per_run.append(dp.get_combiners())

    eval_combiner = Evaluation()
    eval_combiner.set_metrics(*perf_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    if cr:
        eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, y_comb)
    else:
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

    ensemble_mean_score = np.mean([t[1] for t in classifiers_performance_tuples])
    ensemble_mean_scores.append(ensemble_mean_score)

    classifier_max_mean_confidence = classifiers_mean_confidence_tuples[0][1]
    classifier_max_mean_confidences.append(classifier_max_mean_confidence)

    combiners_max_score = eval_combiner.get_top_n_instances()[0][1]
    combiners_max_scores.append(combiners_max_score)

    performance_improvement = combiners_max_score - classifier_max_score
    performance_improvements.append(performance_improvement)

    classifier_score_std = np.std([t[1] for t in classifiers_performance_tuples])
    classifier_score_stds.append(classifier_score_std)

    best_combiners_per_run.append(eval_combiner.get_top_n_instances()[0][0])

    best_combiner_perf_tuples_per_run.append(eval_combiner.get_top_instances())

    if not cr:
        ensemble_diversity_correlation_scores.append(pairwise_correlation(y_ensemble_test, y_test))
        ensemble_diversity_q_statistic_scores.append(pairwise_q_statistic(y_ensemble_test, y_test))
        ensemble_diversity_kappa_statistic.append(pairwise_kappa_statistic(y_ensemble_test, y_test))
        ensemble_diversity_disagreement.append(pairwise_disagreement(y_ensemble_test, y_test))
        ensemble_diversity_double_fault.append(pairwise_double_fault(y_ensemble_test, y_test))

        ensemble_diversity_cohens_kappa_scores.append(pairwise_cohens_kappa(y_ensemble_test))
        ensemble_pairwise_euclidean_distance.append(pairwise_euclidean_distance(y_ensemble_test))

    combiners_runtime_matrix = eval_combiner.get_runtime_matrix()
    combiners_runtime_run_matrices.append(combiners_runtime_matrix)


# === Plot properties ==================================================================================================
meanprops = dict(markerfacecolor='black', markeredgecolor='white')
plt.rcParams.update({'font.size': 13})


def extend_x_ticks_upper_bound(plot):
    x_ticks = plot.xticks()[0].tolist()
    step = x_ticks[-1] - x_ticks[-2]
    x_ticks.append(x_ticks[-1] + step)
    return x_ticks


def extend_y_ticks_upper_bound(plot):
    y_ticks = plot.yticks()[0].tolist()
    step = y_ticks[-1] - y_ticks[-2]
    y_ticks.append(y_ticks[-1] + step)
    return y_ticks


# === Fusion methods comparison ========================================================================================

# --- Fusion methods performance comparison ----------------------------------------------------------------------------
reduced_combiners_performances = {}
for perf_tuples in combiners_performance_run_tuples:  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in reduced_combiners_performances:  # create a score list if non-existent
            reduced_combiners_performances[combiner] = []
        reduced_combiners_performances[combiner].append(t[1])

combiners = [comb for comb in reduced_combiners_performances.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performances = [reduced_combiners_performances[c] for c in combiners]

eval_dict['classifiers_mean_performance'] = np.mean(classifier_max_scores)
eval_dict['classifiers_median_performance'] = np.median(classifier_max_scores)
eval_dict['combiners'] = combiners_names
eval_dict['combiners_mean_performances'] = \
    [t for t in zip(combiners_names, [np.mean(reduced_combiners_performances[c]) for c in combiners])]
eval_dict['combiners_median_performances'] = \
    [t for t in zip(combiners_names, [np.median(reduced_combiners_performances[c]) for c in combiners])]


# Add control (best classifier performance)
combiners_performances.insert(0, classifier_max_scores)
combiners_names.insert(0, 'Kontrolle')

plt.figure(figsize=(10, 4.8))
bp = plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops, patch_artist=True)
for box in bp['boxes']:
    box.set_facecolor('white')
# plt.title("Performanzvergleich der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Trefferquote", fontweight='bold', labelpad=15)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
for y_tick in plt.yticks()[0].tolist():
    plt.axhline(y_tick, color='#bbbbbb', linewidth=0.5, linestyle='--', zorder=-1)
plt.tight_layout()
save(plt, "010_box_plot_combiner_control_comparison", eval_id)
plt.close()

# --- Fusion methods performance improvement comparison ----------------------------------------------------------------
combiner_wise_perf_differences = {}
for i, perf_tuples in enumerate(combiners_performance_run_tuples):  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in combiner_wise_perf_differences:  # create a score list if non-existent
            combiner_wise_perf_differences[combiner] = []
        combiner_wise_perf_differences[combiner].append(t[1] - classifier_max_scores[i])

combiners = [comb for comb in combiner_wise_perf_differences.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_improvements = [combiner_wise_perf_differences[c] for c in combiners]

eval_dict['combiners_mean_performance_improvement'] = \
    [t for t in zip(combiners_names, [np.mean(combiner_wise_perf_differences[c]) for c in combiners])]
eval_dict['combiners_median_performance_improvement'] = \
    [t for t in zip(combiners_names, [np.median(combiner_wise_perf_differences[c]) for c in combiners])]
eval_dict['combiners_min_performance_improvement'] = \
    [t for t in zip(combiners_names, [np.min(combiner_wise_perf_differences[c]) for c in combiners])]
eval_dict['combiners_max_performance_improvement'] = \
    [t for t in zip(combiners_names, [np.max(combiner_wise_perf_differences[c]) for c in combiners])]

plt.figure(figsize=(10, 4.8))
bp = plt.boxplot(combiners_improvements, showmeans=True, meanprops=meanprops, patch_artist=True)
for box in bp['boxes']:
    box.set_facecolor('white')
# plt.title("Performanzvergleich der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Differenz (Trefferquote)", fontweight='bold', labelpad=15)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
for y_tick in plt.yticks()[0].tolist():
    plt.axhline(y_tick, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
save(plt, "011_box_plot_combiner_perf_improvements", eval_id)
plt.close()


# --- Fusion methods performance improvement comparison (positive means only) ------------------------------------------
combiner_wise_perf_differences = {}
for i, perf_tuples in enumerate(combiners_performance_run_tuples):  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in combiner_wise_perf_differences:  # create a score list if non-existent
            combiner_wise_perf_differences[combiner] = []
        combiner_wise_perf_differences[combiner].append(t[1] - classifier_max_scores[i])

# filter out combiners with negative mean performance difference.
combiners = [comb for comb in combiner_wise_perf_differences.keys()
             if np.mean(combiner_wise_perf_differences[comb]) > 0]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_improvements = [combiner_wise_perf_differences[c] for c in combiners]

plt.figure(figsize=(10, 4.8))
bp = plt.boxplot(combiners_improvements, showmeans=True, meanprops=meanprops, patch_artist=True)
for box in bp['boxes']:
    box.set_facecolor('white')
# plt.title("Performanzvergleich der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Differenz (Trefferquote)", fontweight='bold', labelpad=15)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
for y_tick in plt.yticks()[0].tolist():
    plt.axhline(y_tick, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
save(plt, "011_box_plot_combiner_perf_improvements_positive_means", eval_id)
plt.close()


# --- Fusion methods confidence comparison -----------------------------------------------------------------------------
reduced_combiners_mean_confidences = {}
for perf_tuples in combiners_mean_confidence_run_tuples:  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in reduced_combiners_mean_confidences:  # create a score list if non-existent
            reduced_combiners_mean_confidences[combiner] = []
        reduced_combiners_mean_confidences[combiner].append(t[1])

combiners = [comb for comb in reduced_combiners_mean_confidences.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performances = [reduced_combiners_mean_confidences[c] for c in combiners]

eval_dict['classifiers_mean_confidence'] = np.mean(classifier_max_mean_confidences)
eval_dict['classifiers_median_confidence'] = np.median(classifier_max_mean_confidences)
eval_dict['combiners_mean_confidence'] = \
    [t for t in zip(combiners_names, [np.mean(reduced_combiners_mean_confidences[c]) for c in combiners])]
eval_dict['combiners_median_confidence'] = \
    [t for t in zip(combiners_names, [np.median(reduced_combiners_mean_confidences[c]) for c in combiners])]

# Add control (best classifier confidence)
combiners_performances.insert(0, classifier_max_mean_confidences)
combiners_names.insert(0, 'Kontrolle')

plt.figure(figsize=(10, 4.8))
bp = plt.boxplot(combiners_performances, showmeans=True, meanprops=meanprops, patch_artist=True)
for box in bp['boxes']:
    box.set_facecolor('white')
# plt.title("Performanzvergleich der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Mittlere Konfidenz", fontweight='bold', labelpad=15)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
for y_tick in plt.yticks()[0].tolist():
    plt.axhline(y_tick, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
save(plt, "021_box_plot_combiner_control_comparison_mean_confidence", eval_id)
plt.close()

# --- Fusion methods confidence improvement comparison -----------------------------------------------------------------
combiner_wise_conf_differences = {}
for i, perf_tuples in enumerate(combiners_mean_confidence_run_tuples):  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in combiner_wise_conf_differences:  # create a score list if non-existent
            combiner_wise_conf_differences[combiner] = []
        combiner_wise_conf_differences[combiner].append(t[1] - classifier_max_mean_confidences[i])

combiners = [comb for comb in combiner_wise_conf_differences.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_improvements = [combiner_wise_conf_differences[c] for c in combiners]

eval_dict['combiners_mean_mean_confidence_improvement'] = \
    [t for t in zip(combiners_names, [np.mean(combiner_wise_conf_differences[c]) for c in combiners])]
eval_dict['combiners_median_mean_confidence_improvement'] = \
    [t for t in zip(combiners_names, [np.median(combiner_wise_conf_differences[c]) for c in combiners])]
eval_dict['combiners_min_mean_confidence_improvement'] = \
    [t for t in zip(combiners_names, [np.min(combiner_wise_conf_differences[c]) for c in combiners])]
eval_dict['combiners_max_mean_confidence_improvement'] = \
    [t for t in zip(combiners_names, [np.max(combiner_wise_conf_differences[c]) for c in combiners])]


plt.figure(figsize=(10, 4.8))
bp = plt.boxplot(combiners_improvements, showmeans=True, meanprops=meanprops, patch_artist=True)
for box in bp['boxes']:
    box.set_facecolor('white')
# plt.title("Performanzvergleich der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.ylabel("Differenz (Mittlere Konfidenz)", fontweight='bold', labelpad=15)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(combiners_names)+1), combiners_names)
for y_tick in plt.yticks()[0].tolist():
    plt.axhline(y_tick, color='grey', linewidth=0.5, linestyle='--')
plt.tight_layout()
save(plt, "022_box_plot_combiner_confidence_improvements", eval_id)
plt.close()

# === Performance comparison (Ensemble/Framework) ======================================================================

eval_dict['classifier_max_scores_overall_mean'] = np.mean(classifier_max_scores)
eval_dict['combiners_max_scores_overall_mean'] = np.mean(combiners_max_scores)
eval_dict['combiners_max_scores_overall_min'] = np.min(combiners_max_scores)
eval_dict['combiners_max_scores_overall_max'] = np.max(combiners_max_scores)

eval_dict['combiners_performance_improvements_mean'] = np.mean(performance_improvements)
eval_dict['combiners_performance_improvements_min'] = np.min(performance_improvements)
eval_dict['combiners_performance_improvements_max'] = np.max(performance_improvements)

plt.figure()
plt.boxplot([classifier_max_scores, combiners_max_scores], showmeans=True, meanprops=meanprops)
# plt.title("Performanzvergleich (" + str(n_runs) + " runs)")
plt.ylabel("Max. Trefferquote", fontweight='bold', labelpad=15)
plt.xticks([1, 2], ['Ensemble', 'Framework'])
plt.tight_layout()
save(plt, "030_box_plot_max_performance_comparison", eval_id)
plt.close()

# --- Performance improvement by Framework -----------------------------------------------------------------------------
plt.figure()
plt.boxplot(performance_improvements, showmeans=True, meanprops=meanprops)
# plt.title("Performanzverbesserung (" + str(n_runs) + " runs)")
plt.ylabel("Trefferquote (Differenz)", fontweight='bold', labelpad=15)
plt.xticks([1], ['Framework'])
plt.tight_layout()
save(plt, "031_box_plot_performance_improvement", eval_id)
plt.close()

# --- Positive performance improvement by each fusion method -----------------------------------------------------------
reduced_combiners_performance_differences = {}
for perf_tuples, cls_max_score in zip(combiners_performance_run_tuples, classifier_max_scores):  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        combiner_score = t[1]
        if combiner not in reduced_combiners_performance_differences:  # create a score list if non-existent
            reduced_combiners_performance_differences[combiner] = []
        if combiner_score - cls_max_score > 0:
            reduced_combiners_performance_differences[combiner].append(combiner_score - cls_max_score)
        else:
            reduced_combiners_performance_differences[combiner].append(0)

combiners = [comb for comb in reduced_combiners_performance_differences.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
combiners_performance_improvements = np.around(
    [np.mean(reduced_combiners_performance_differences[c]) for c in combiners], 4)

# filter out combiners with 0 performance improvement
non_improving_comb_indices = np.where(combiners_performance_improvements == 0)[0]
combiners = np.delete(np.array(combiners), non_improving_comb_indices)
combiners_names = np.delete(np.array(combiners_names), non_improving_comb_indices)
combiners_performance_improvements = np.delete(combiners_performance_improvements, non_improving_comb_indices)


df = pd.DataFrame({'combiners_names': combiners_names,
                   'combiners_performance_improvements': combiners_performance_improvements})
df_sorted = df.sort_values('combiners_performance_improvements', ascending=False)

plt.figure()
fig, ax = plt.subplots()
# p = ax.bar(combiners_names, combiners_performance_improvements, yerr=combiners_performance_improvements_stds)
# p = ax.barh(combiners_names, combiners_performance_improvements, height=0.2, color='black')
p = plt.barh('combiners_names', 'combiners_performance_improvements', data=df_sorted, height=0.2)
ax.bar_label(p, padding=3)
plt.xlabel("Mittlere positive Performanzdifferenz (Trefferquote)", fontweight='bold', labelpad=15)
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.tight_layout()
save(plt, "032_mean_positive_performance_difference_per_fusion_method", eval_id)
plt.close()

# === Performance Profiles =============================================================================================
plt.rcParams.update({'font.size': 12})
reduced_combiners_performances = {}
for perf_tuples in combiners_performance_run_tuples:  # reduce
    for t in perf_tuples:
        combiner = type(t[0])
        if combiner not in reduced_combiners_performances:  # create a score list if non-existent
            reduced_combiners_performances[combiner] = []
        reduced_combiners_performances[combiner].append(t[1])

combiners = [comb for comb in reduced_combiners_performances.keys()]
combiners_names = [c.SHORT_NAME for c in combiners]
# combiners_performances = [reduced_combiners_performances[c] for c in combiners]

plt.figure(figsize=(10, 4.8))

n_x_cells = 3
n_y_cells = int(len(combiners)/n_x_cells) if len(combiners) % n_x_cells == 0 else int(len(combiners)/n_x_cells) + 1
fig, axs = plt.subplots(n_y_cells, n_x_cells, sharex='all', sharey='all', figsize=(10, 10))

for k, comb in enumerate(combiners):
    i = int(k / 3)
    j = k % 3
    axs[i, j].scatter(classifier_max_scores, reduced_combiners_performances[comb],
                      s=25, c='black', marker="x", linewidth=.6)
    # axs[i, j].scatter(ensemble_mean_scores, reduced_combiners_performances[comb],
    #                   s=25, marker="o", linewidth=1, facecolor='none', edgecolors='black')
    axs[i, j].plot([0, 1], [0, 1], linewidth=0.5, c='#8a8a8a')
    axs[i, j].set_title(combiners_names[k])

for k in range(len(combiners), n_x_cells * n_y_cells):
    fig.delaxes(axs[int(k / n_x_cells), k % n_x_cells])

plt.setp(axs[-1, :], xlabel='Trefferquote (Ensemble)')
plt.setp(axs[:, 0], ylabel='Trefferquote (Fusion)')

plt.tight_layout()
save(plt, "040_performance_profiles", eval_id)
plt.close()


# === Diversity -- Framework-Performanz ===============================================================================
plt.rcParams.update({'font.size': 13})
if not cr:
    plt.figure()
    plt.plot(ensemble_diversity_kappa_statistic, combiners_max_scores, 'g^')
    plt.xlabel("Diversität (Kappa-Statistik)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_00_div_cohens_kappa2__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_correlation_scores, combiners_max_scores, 'bs')
    plt.xlabel("Diversität (Korrelation)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_01_div_correlation__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_q_statistic_scores, combiners_max_scores, 'g^')
    plt.xlabel("Diversität (Q-Statistik)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_02_div_q_stat__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_disagreement, combiners_max_scores, 'mv')
    plt.xlabel("Diversität (Disagreement)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_03_div_disagreement__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_double_fault, combiners_max_scores, 'rH')
    plt.xlabel("Diversität (Double Fault)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_04_div_double_fault__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_cohens_kappa_scores, combiners_max_scores, 'ro')
    plt.xlabel("Diversität (Cohen's Kappa)", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_05_div_cohens_kappa__framework_performance", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_pairwise_euclidean_distance, combiners_max_scores, 'gD')
    plt.xlabel("Mittlere paarweise euklidische Distanz", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "100_data_plot_06_euclidean_distance__framework_performance", eval_id)
    plt.close()

    # === Diversity -- Performanzverbesserung =========================================================================

    plt.figure()
    plt.plot(ensemble_diversity_kappa_statistic, performance_improvements, 'ro')
    plt.xlabel("Diversität (Kappa statistic)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_10_div_cohens_kappa2__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_correlation_scores, performance_improvements, 'bs')
    plt.xlabel("Diversität (Correlation)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_11_div_correlation__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_q_statistic_scores, performance_improvements, 'g^')
    plt.xlabel("Diversität (Q-statistic)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_12_div_q_stat__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_disagreement, performance_improvements, 'mv')
    plt.xlabel("Diversität (Disagreement)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_13_div_disagreement__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_double_fault, performance_improvements, 'rH')
    plt.xlabel("Diversität (Double Fault)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_14_div_double_fault__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_diversity_cohens_kappa_scores, performance_improvements, 'ro')
    plt.xlabel("Diversität (Cohen's Kappa)", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_15_div_cohens_kappa__perf_improvement", eval_id)
    plt.close()

    plt.figure()
    plt.plot(ensemble_pairwise_euclidean_distance, performance_improvements, 'bD')
    plt.xlabel("Mean pairwise Euclidean distance", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "200_data_plot_16_euclidean_distance__perf_improvement", eval_id)
    plt.close()

    # === Diversity - Framework-Performanz - Mean Ensemble Performance ================================================

    mean_classifier_perf_per_run = []
    for perf_tuples in classifiers_performance_run_tuples:
        mean_classifier_perf_per_run.append(np.mean([t[1] for t in perf_tuples]))

    fig, ax = plt.subplots()
    scatter = ax.scatter(ensemble_diversity_correlation_scores, combiners_max_scores, c=mean_classifier_perf_per_run)
    ax.set_xlabel('Diversität (Korrelation)', fontweight='bold', labelpad=15)
    ax.set_ylabel('Framework-Performanz (Trefferquote)', fontweight='bold', labelpad=15)
    fig.colorbar(scatter).set_label("Mittlere Ensemble-Performance (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "300_scatter_plot_cls_mean_acc__framework_performance__diversity_correlation", eval_id)
    plt.close()

    # === Diversity - Performanzverbesserung - Mean Ensemble Performance ==============================================

    fig, ax = plt.subplots()
    scatter = ax.scatter(ensemble_diversity_correlation_scores, performance_improvements, c=mean_classifier_perf_per_run)
    ax.set_xlabel('Diversität (Korrelation)', fontweight='bold', labelpad=15)
    ax.set_ylabel('Performanzverbesserung (Trefferquote)', fontweight='bold', labelpad=15)
    fig.colorbar(scatter).set_label("Mittlere Ensemble-Performance (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "301_scatter_plot_cls_mean_acc__performance_imp__diversity_correlation", eval_id)
    plt.close()

# === Coverage =========================================================================================================
if cr:
    # --- Coverage - Max. scores ---------------------------------------------------------------------------------------
    plt.figure()
    plt.plot(coverage_overlaps, combiners_max_scores, 'bx')
    plt.xlabel("Überdeckungsdichte", fontweight='bold', labelpad=15)
    plt.ylabel("Framework-Performanz (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "050_data_plot_01_cr_overlap__framework_performance", eval_id)
    plt.close()

    # --- Coverage - Improvement ---------------------------------------------------------------------------------------
    plt.figure()
    plt.plot(coverage_overlaps, performance_improvements, 'rx')
    plt.xlabel("Überdeckungsdichte", fontweight='bold', labelpad=15)
    plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
    plt.tight_layout()
    save(plt, "051_data_plot_01_cr_overlap__framework_imp", eval_id)
    plt.close()

# === Combiner Frequencies =============================================================================================

best_combiners__short_names = [c.SHORT_NAME for c in best_combiners_per_run]
unique_best_combiners = np.unique(best_combiners__short_names, return_counts=True)
combiners_names = unique_best_combiners[0]
combiners_frequency = unique_best_combiners[1]

# --- Frequencies of all combiners -------------------------------------------------------------------------------------
df = pd.DataFrame({'combiners_names': combiners_names, 'combiners_frequency': combiners_frequency})
df_sorted = df.sort_values('combiners_frequency')

plt.figure(figsize=(10, 4.8))
# plt.bar(combiners_names, combiners_frequency, color='gray')
plt.bar('combiners_names', 'combiners_frequency', data=df_sorted, color='gray')
plt.title("Auftrittshäufigkeit der besten Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Auftrittsfrequenz", fontweight='bold', labelpad=15)
plt.tight_layout()
# save(plt, "400_combiner_frequency", eval_id)
plt.close()

# --- Frequencies of improving combiners -------------------------------------------------------------------------------
improving_combiners__short_names = []
for i, classifier_max_score in enumerate(classifier_max_scores):
    best_combiner_perf_tuples = best_combiner_perf_tuples_per_run[i]
    if best_combiner_perf_tuples[0][1] > classifier_max_score:
        for t in best_combiner_perf_tuples:
            improving_combiners__short_names.append(t[0].SHORT_NAME)

unique_best_combiners = np.unique(improving_combiners__short_names, return_counts=True)
combiners_names = unique_best_combiners[0]
combiners_frequency = unique_best_combiners[1]

df = pd.DataFrame({'combiners_names': combiners_names, 'combiners_frequency': combiners_frequency})
df_sorted = df.sort_values('combiners_frequency')

plt.figure()
plt.bar('combiners_names', 'combiners_frequency', data=df_sorted, color='gray')
plt.title("Auftrittshäufigkeit verbessernder Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Auftrittsfrequenz", fontweight='bold', labelpad=15)
plt.tight_layout()
# save(plt, "401_improving_combiner_frequency", eval_id)
plt.close()

# --- Frequencies of improving combiners (percentage) ------------------------------------------------------------------
combiners_frequency = [cf/n_runs*100 for cf in combiners_frequency]

df = pd.DataFrame({'combiners_names': combiners_names, 'combiners_frequency': combiners_frequency})
df_sorted = df.sort_values('combiners_frequency', ascending=False)

plt.figure()
bar1 = plt.barh('combiners_names', 'combiners_frequency', data=df_sorted, color='#5a6f9c', height=0.2)
# plt.title("Auftrittshäufigkeit verbessernder Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.xlabel("Auftrittsfrequenz in %", labelpad=15, fontweight='bold')
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "402_improving_combiner_frequency_h", eval_id)
plt.close()

# === Ensemble STD =====================================================================================================

# --- Ensemble STD - Framework-Performanz -----------------------------------------------------------------------------
plt.figure()
plt.plot(classifier_score_stds, combiners_max_scores, 'bx')
plt.xlabel("Ensemble-Standardabweichung (Trefferquote)", fontweight='bold', labelpad=15)
plt.ylabel('Framework-Performanz (Trefferquote)', fontweight='bold', labelpad=15)
plt.tight_layout()
save(plt, "310_scatter_plot_ensemble_std__performance", eval_id)
plt.close()

# --- Ensemble STD - Performanzverbesserung ---------------------------------------------------------------------------
plt.figure()
plt.plot(classifier_score_stds, performance_improvements, 'rx')
plt.xlabel("Ensemble-Standardabweichung (Trefferquote)", fontweight='bold', labelpad=15)
plt.ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
plt.tight_layout()
save(plt, "311_scatter_plot_ensemble_std__performance_imp", eval_id)
plt.close()

# --- Ensemble STD - Framework-Performanz - Mean Ensemble Performance -------------------------------------------------
mean_classifier_perf_per_run = []
for perf_tuples in classifiers_performance_run_tuples:
    mean_classifier_perf_per_run.append(np.mean([t[1] for t in perf_tuples]))

fig, ax = plt.subplots()
scatter = ax.scatter(classifier_score_stds, combiners_max_scores, c=mean_classifier_perf_per_run)
ax.set_xlabel("Ensemble-Standardabweichung (Trefferquote)", fontweight='bold', labelpad=15)
ax.set_ylabel('Framework-Performanz (Trefferquote)', fontweight='bold', labelpad=15)
fig.colorbar(scatter).set_label("Ensemble Mean Performance (Trefferquote)", fontweight='bold', labelpad=15)
plt.tight_layout()
save(plt, "320_scatter_plot_ensemble_std__performance__mean_ensemble_performance", eval_id)
plt.close()

# --- Ensemble STD - Performanzverbesserung - Mean Ensemble Performance -----------------------------------------------
fig, ax = plt.subplots()
scatter = ax.scatter(classifier_score_stds, performance_improvements, c=mean_classifier_perf_per_run)
ax.set_xlabel("Ensemble-Standardabweichung (Trefferquote)", fontweight='bold', labelpad=15)
ax.set_ylabel("Performanzverbesserung (Trefferquote)", fontweight='bold', labelpad=15)
fig.colorbar(scatter).set_label("Ensemble Mean Performance (Trefferquote)", fontweight='bold', labelpad=15)
plt.tight_layout()
save(plt, "321_scatter_plot_ensemble_std__performance_imp__mean_ensemble_performance", eval_id)
plt.close()


# === Comparison between utility and trainable combiners ===============================================================


# === Combiner runtimes ================================================================================================

runtime_tensor = np.zeros((len(combiners_runtime_run_matrices), len(combiners_runtime_run_matrices[0]), 2))

for i, runtime_matrix in enumerate(combiners_runtime_run_matrices):
    runtime_tensor[i] = runtime_matrix

runtime_tensor = np.nan_to_num(runtime_tensor)

mean_runtime_matrix = np.nanmean(runtime_tensor, axis=0)
combiners_train_mean_runtimes = np.around(mean_runtime_matrix[:, 0], 4)
combiners_combine_mean_runtimes = np.around(mean_runtime_matrix[:, 1], 4)
combiners_names = [c.SHORT_NAME for c in combiners_per_run[0]]

# --- Mean train runtimes ----------------------------------------------------------------------------------------------
non_zero_indexes = np.nonzero(combiners_train_mean_runtimes)[0]
combiners_train_mean_non_zero_runtimes = combiners_train_mean_runtimes[non_zero_indexes]
combiners_non_zero_names = [combiners_names[i] for i in non_zero_indexes]

eval_dict['combiners_train_mean_non_zero_runtimes'] = list(combiners_non_zero_names)
eval_dict['combiners_train_runtime_non_zero_names'] = list(combiners_train_mean_non_zero_runtimes.tolist())

# remove outliers
max_index = np.argmax(combiners_train_mean_non_zero_runtimes)
combiners_train_mean_non_zero_runtimes = np.delete(combiners_train_mean_non_zero_runtimes, max_index)
del combiners_non_zero_names[max_index]

df = pd.DataFrame({'combiners_non_zero_names': combiners_non_zero_names,
                   'combiners_train_mean_non_zero_runtimes': combiners_train_mean_non_zero_runtimes})
df_sorted = df.sort_values('combiners_train_mean_non_zero_runtimes')
plt.figure(figsize=(10, 4.8))
# plt.bar(combiners_non_zero_names, combiners_train_mean_non_zero_runtimes, color='#93c6ed')
bar1 = plt.bar('combiners_non_zero_names', 'combiners_train_mean_non_zero_runtimes', data=df_sorted, color='#93c6ed',
               width=.75)
# plt.title("Mittlere Trainingslaufzeit der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.yticks(extend_y_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z90_train_runtime_comparison", eval_id)
plt.close()

# --- horizontal bar plot
plt.figure()
bar1 = plt.barh('combiners_non_zero_names', 'combiners_train_mean_non_zero_runtimes', data=df_sorted, color='#b5670e',
                height=.2)
plt.xlabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z91_train_runtime_comparison_h", eval_id)
plt.close()

# --- Mean combine runtimes --------------------------------------------------------------------------------------------
df = pd.DataFrame({'combiners_names': combiners_names,
                   'combiners_combine_mean_runtimes': combiners_combine_mean_runtimes})
df_sorted = df.sort_values('combiners_combine_mean_runtimes')

eval_dict['combiners_combine_runtime_names'] = combiners_names
eval_dict['combiners_combine_mean_runtimes'] = combiners_combine_mean_runtimes.tolist()

plt.figure(figsize=(10, 4.8))
# plt.bar(combiners_names, combiners_combine_mean_runtimes, color='#006aba')
bar1 = plt.bar('combiners_names', 'combiners_combine_mean_runtimes', data=df_sorted, color='#006aba')
# plt.title("Mittlere Fusionslaufzeit der Fusionsmethoden (" + str(n_runs) + " Läufe)")
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.yticks(extend_y_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z92_combine_runtime_comparison", eval_id)
plt.close()

# --- horizontal bar plot
plt.figure()
bar1 = plt.barh('combiners_names', 'combiners_combine_mean_runtimes', data=df_sorted, color='#915006', height=.35)
plt.xlabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z93_combine_runtime_comparison_h", eval_id)
plt.close()

# === Data tables ======================================================================================================

dump_data_as_txt(eval_dict, '_stats', eval_id)

save_evaluator(__file__, eval_id)
print("Evaluation", eval_id, "done.")
