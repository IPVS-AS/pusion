import time
import warnings

import matplotlib.pyplot as plt

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *

warnings.filterwarnings('error')  # halt on warning

eval_id = time.strftime("%Y%m%d-%H%M%S")

perf_metrics = (p.PerformanceMetric.ACCURACY, p.PerformanceMetric.MICRO_F1_SCORE, p.PerformanceMetric.MEAN_CONFIDENCE)


# data1 = load_native_files_as_data(['datasets/ensembles_generated_multiclass_classification.pickle'])
# data2 = load_native_files_as_data(['datasets/ensembles_generated_multilabel_classification.pickle'])
# data3 = load_native_files_as_data(['datasets/ensembles_generated_cr_multiclass_classification.pickle'])
# data4 = load_native_files_as_data(['datasets/ensembles_generated_cr_multilabel_classification.pickle'])

data = load_pickle_files_as_data(['datasets/ensembles_generated_multiclass_classification.pickle'])[0]

# Flag for complementary-redundant decision outputs
cr = False

# Ensemble data
ensembles = data['ensembles']
n_classes = data['n_classes']
n_samples = data['n_samples']
random_state = data['random_state']

# ----------------------------------------------------------------------------------------------------------------------

ensemble_wise_type = []
ensemble_wise_accuracies = []
ensemble_wise_max_accuracy = []
ensemble_wise_mean_accuracy = []
ensemble_wise_max_confidence = []
ensemble_wise_combiner_accuracies = []
ensemble_wise_accuracy_improvement = []
ensemble_wise_combiner_confidences = []
ensemble_wise_confidence_improvement = []

np.random.seed(random_state)

for i in ensembles:
    ensemble = ensembles[i]

    y_ensemble_valid = ensemble['y_ensemble_valid']
    y_valid = ensemble['y_valid']
    y_ensemble_test = ensemble['y_ensemble_test']
    y_test = ensemble['y_test']

    print("============== Ensemble ================")
    eval_ensemble = Evaluation(*perf_metrics)
    eval_ensemble.set_instances(ensemble['classifiers'])
    if cr:
        eval_ensemble.set_instances('Ensemble')
        eval_ensemble.evaluate_cr_decision_outputs(y_test, y_ensemble_test, ensemble['coverage'])
    else:
        eval_ensemble.set_instances(ensemble['classifiers'])
        eval_ensemble.evaluate(y_test, y_ensemble_test)
    print(eval_ensemble.get_report())

    print("=========== GenericCombiner ============")
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.set_parallel(False)
    if cr:
        dp.set_coverage(ensemble['coverage'])
    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

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
    print("----------------------------------------")
    print()

    # ------------------------------------------------------------------------------------------------------------------
    ensemble_wise_type.append(ensemble['ensemble_type'])

    ensemble_max_accuracy = eval_ensemble.get_top_n_instances(n=1, metric=p.PerformanceMetric.ACCURACY)[0][1]
    ensemble_wise_max_accuracy.append(ensemble_max_accuracy)

    ensemble_max_confidence = eval_ensemble.get_top_n_instances(n=1, metric=p.PerformanceMetric.MEAN_CONFIDENCE)[0][1]
    ensemble_wise_max_confidence.append(ensemble_max_confidence)

    ensemble_accuracies = [t[1] for t in eval_ensemble.get_top_n_instances(metric=p.PerformanceMetric.ACCURACY)]
    ensemble_wise_accuracies.append(ensemble_accuracies)

    ensemble_mean_accuracy = np.mean(ensemble_accuracies)
    ensemble_wise_mean_accuracy.append(ensemble_mean_accuracy)

    combiner_accuracy_tuples = eval_combiner.get_instance_performance_tuples(p.PerformanceMetric.ACCURACY)
    ensemble_wise_combiner_accuracies.append(combiner_accuracy_tuples)

    combiner_accuracies = np.array([t[1] for t in combiner_accuracy_tuples])
    ensemble_wise_accuracy_improvement.append(combiner_accuracies-ensemble_max_accuracy)

    combiner_confidence_tuples = eval_combiner.get_instance_performance_tuples(p.PerformanceMetric.MEAN_CONFIDENCE)
    ensemble_wise_combiner_confidences.append(combiner_confidence_tuples)

    combiner_confidences = np.array([t[1] for t in combiner_confidence_tuples])
    ensemble_wise_confidence_improvement.append(combiner_confidences-ensemble_max_confidence)


# === Plots ============================================================================================================
meanprops = dict(markerfacecolor='black', markeredgecolor='white')


def extend_y_ticks(plot):
    y_ticks = plot.yticks()[0].tolist()
    step = y_ticks[-1] - y_ticks[-2]
    y_ticks.insert(0, y_ticks[0] - step)
    y_ticks.append(y_ticks[-1] + step)
    return y_ticks


# --- Ensemble max. accuracy -------------------------------------------------------------------------------------------
plt.figure()
plt.bar(ensemble_wise_type, ensemble_wise_max_accuracy, color='#006aba')
plt.xlabel("Ensembles", fontweight='bold', labelpad=15)
plt.ylabel("Max. Trefferquote", fontweight='bold', labelpad=15)
plt.tight_layout()
# save(plt, "000_ensemble_max_accuracy", eval_id)
plt.close()

# --- Ensemble accuracies ----------------------------------------------------------------------------------------------
plt.figure()
plt.boxplot(ensemble_wise_accuracies, showmeans=True, meanprops=meanprops)
plt.xlabel("Ensembles", fontweight='bold', labelpad=15)
plt.ylabel("Trefferquote", fontweight='bold', labelpad=15)
plt.xticks(np.arange(1, len(ensemble_wise_type) + 1), ensemble_wise_type)
plt.tight_layout()
save(plt, "001_ensemble_accuracies", eval_id)
plt.close()

# --- Combiner accuracies per ensemble ---------------------------------------------------------------------------------
plt.figure(figsize=(12, 5.5))
bar1 = np.around([t[1] for t in ensemble_wise_combiner_accuracies[0]], 3)
bar2 = np.around([t[1] for t in ensemble_wise_combiner_accuracies[1]], 3)
bar3 = np.around([t[1] for t in ensemble_wise_combiner_accuracies[2]], 3)

barWidth = 0.25
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label=ensemble_wise_type[0] + "-Ensemble")
rect2 = plt.bar(r2, bar2, color='#5a6f9c', width=barWidth, edgecolor='white', label=ensemble_wise_type[1] + "-Ensemble")
rect3 = plt.bar(r3, bar3, color='#9ab2e6', width=barWidth, edgecolor='white', label=ensemble_wise_type[2] + "-Ensemble")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.ylabel('Trefferquote', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [t[0].SHORT_NAME for t in ensemble_wise_combiner_accuracies[0]])
plt.yticks(np.arange(0, 1.1, .1))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "002_combiner_accuracies_grouped", eval_id)
plt.close()


# --- Combiner accuracy improvement per ensemble -----------------------------------------------------------------------
plt.figure(figsize=(12, 5.5))
bar1 = np.around(ensemble_wise_accuracy_improvement[0], 3)
bar2 = np.around(ensemble_wise_accuracy_improvement[1], 3)
bar3 = np.around(ensemble_wise_accuracy_improvement[2], 3)

barWidth = 0.25
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label=ensemble_wise_type[0] + "-Ensemble")
rect2 = plt.bar(r2, bar2, color='#5a6f9c', width=barWidth, edgecolor='white', label=ensemble_wise_type[1] + "-Ensemble")
rect3 = plt.bar(r3, bar3, color='#9ab2e6', width=barWidth, edgecolor='white', label=ensemble_wise_type[2] + "-Ensemble")

plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.ylabel('Differenz in der Trefferquote', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [t[0].SHORT_NAME for t in ensemble_wise_combiner_accuracies[0]])
plt.yticks(extend_y_ticks(plt))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "003_combiner_improvements_grouped", eval_id)
plt.close()


# --- Combiner confidences per ensemble --------------------------------------------------------------------------------
plt.figure(figsize=(12, 5.5))
bar1 = np.around([t[1] for t in ensemble_wise_combiner_confidences[0]], 3)
bar2 = np.around([t[1] for t in ensemble_wise_combiner_confidences[1]], 3)
bar3 = np.around([t[1] for t in ensemble_wise_combiner_confidences[2]], 3)

barWidth = 0.25
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label=ensemble_wise_type[0] + "-Ensemble")
rect2 = plt.bar(r2, bar2, color='#5a6f9c', width=barWidth, edgecolor='white', label=ensemble_wise_type[1] + "-Ensemble")
rect3 = plt.bar(r3, bar3, color='#9ab2e6', width=barWidth, edgecolor='white', label=ensemble_wise_type[2] + "-Ensemble")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.ylabel('Mittlere Konfidenz', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [t[0].SHORT_NAME for t in ensemble_wise_combiner_confidences[0]])
plt.yticks(np.arange(0, 1.1, .1))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "004_combiner_confidences_grouped", eval_id)
plt.close()


# --- Combiner confidence improvement per ensemble ---------------------------------------------------------------------
plt.figure(figsize=(12, 5.5))
bar1 = np.around(ensemble_wise_confidence_improvement[0], 3)
bar2 = np.around(ensemble_wise_confidence_improvement[1], 3)
bar3 = np.around(ensemble_wise_confidence_improvement[2], 3)

barWidth = 0.25
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label=ensemble_wise_type[0] + "-Ensemble")
rect2 = plt.bar(r2, bar2, color='#5a6f9c', width=barWidth, edgecolor='white', label=ensemble_wise_type[1] + "-Ensemble")
rect3 = plt.bar(r3, bar3, color='#9ab2e6', width=barWidth, edgecolor='white', label=ensemble_wise_type[2] + "-Ensemble")

plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)
plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.ylabel('Differenz in der mittleren Konfidenz', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [t[0].SHORT_NAME for t in ensemble_wise_combiner_confidences[0]])
plt.yticks(extend_y_ticks(plt))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "005_combiner_improvements_grouped", eval_id)
plt.close()

# ======================================================================================================================
save_evaluator(__file__, eval_id)
print("Evaluation", eval_id, "done.")
