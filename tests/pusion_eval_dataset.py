import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.input_output.file_input_output import *
from pusion.util.generator import split_into_train_and_validation_data
from pusion.util.transformer import *

warnings.filterwarnings('error')  # halt on warning

eval_id = time.strftime("%Y%m%d-%H%M%S")

random_state = 1

dataset_files = [
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep24_1.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep04_2.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep24_3.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_4.pickle',
    'datasets/Time-SE-ResNet_lr0.01_bs128_ep70_5.pickle',
    'datasets/IndRnn_Classification_lr0.001_bs128_ep35_1.pickle',
    'datasets/IndRnn_Classification_lr0.001_bs128_ep35_2.pickle',
    'datasets/IndRnn_Classification_lr0.001_bs128_ep35_3.pickle',
]

data = load_native_files_as_data(dataset_files)

decision_outputs = [
    # data[0]['Y_predictions'],
    # data[1]['Y_predictions'],
    # data[2]['Y_predictions'],
    # data[3]['Y_predictions'],
    # data[4]['Y_predictions'],
    # data[5]['Y_predictions'],
    data[6]['Y_predictions'],
    data[7]['Y_predictions'],
]

true_assignments = np.array(data[6]['Y_test'])

coverage = [
    [0,  1,  2,  3],
    [0,  1,  2,  3],
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15],
    [0,  8]
]

cr = False

np.random.seed(random_state)

y_ensemble_valid, y_valid, y_ensemble_test, y_test = \
    split_into_train_and_validation_data(decision_outputs, true_assignments, validation_size=.75)

eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.F1_SCORE,
    p.PerformanceMetric.MEAN_CONFIDENCE,
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
y_ensemble_test = multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test)
if cr:
    eval_classifiers.set_instances(['Ensemble'])
    eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
else:
    eval_classifiers.set_instances([('Classifier ' + str(i)) for i in range(len(decision_outputs))])
    eval_classifiers.evaluate(y_test, y_ensemble_test)

print(eval_classifiers.get_report())

# ---- GenericCombiner -------------------------------------------------------------------------------------------------
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
if cr:
    dp.set_coverage(coverage)
dp.train(y_ensemble_valid, y_valid)
dp.combine(y_ensemble_test)

# ---- Evaluate all combiners
print("========== GenericCombiner ===========")
eval_combiner = Evaluation(*eval_metrics)
eval_combiner.set_instances(dp.get_combiners())
eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
multi_comb_decision_outputs = multiclass_prediction_tensor_to_decision_tensor(dp.get_multi_combiner_decision_output())
if cr:
    eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, multi_comb_decision_outputs)
else:
    eval_combiner.evaluate(y_test, multi_comb_decision_outputs)
print(eval_combiner.get_report())


# === Plots ============================================================================================================
meanprops = dict(markerfacecolor='black', markeredgecolor='white')


def extend_y_ticks(plot):
    y_ticks = plot.yticks()[0].tolist()
    step = y_ticks[-1] - y_ticks[-2]
    y_ticks.insert(0, y_ticks[0] - step)
    y_ticks.append(y_ticks[-1] + step)
    return y_ticks


# --- Ensemble performance ---------------------------------------------------------------------------------------------
classifiers_accuracies = [t[1] for t in eval_classifiers.get_instance_performance_tuples(p.PerformanceMetric.ACCURACY)]
classifiers_f1_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(p.PerformanceMetric.F1_SCORE)]
classifiers_mean_confidences = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MEAN_CONFIDENCE)]

bar1 = np.around(classifiers_accuracies, 3)
bar2 = np.around(classifiers_f1_scores, 3)
bar3 = np.around(classifiers_mean_confidences, 3)

barWidth = 0.2
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure()
rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#a87f52', width=barWidth, edgecolor='white', label="F1-Score")
rect3 = plt.bar(r3, bar3, color='#52a859', width=barWidth, edgecolor='white', label="Mittlere Konfidenz")

plt.xlabel('Ensemble', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [str(instance) for instance in eval_classifiers.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung', fontweight='bold', labelpad=15)
plt.yticks(np.arange(0, 1.1, .1))
plt.ylim((0, 1.2))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "100_classifier_scores_grouped", eval_id)
plt.close()


# --- Combiners performance --------------------------------------------------------------------------------------------
combiners_accuracies = [t[1] for t in eval_combiner.get_instance_performance_tuples(p.PerformanceMetric.ACCURACY)]
combiners_f1_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(p.PerformanceMetric.F1_SCORE)]
combiners_mean_confidences = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MEAN_CONFIDENCE)]

bar1 = np.around(combiners_accuracies, 3)
bar2 = np.around(combiners_f1_scores, 3)
bar3 = np.around(combiners_mean_confidences, 3)

barWidth = 0.2
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(12, 5.5))
rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#a87f52', width=barWidth, edgecolor='white', label="F1-Score")
rect3 = plt.bar(r3, bar3, color='#52a859', width=barWidth, edgecolor='white', label="Mittlere Konfidenz")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [comb.SHORT_NAME for comb in eval_combiner.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung', fontweight='bold', labelpad=15)
plt.yticks(np.arange(0, 1.1, .1))
plt.ylim((0, 1.2))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "101_combiner_scores_grouped", eval_id)
plt.close()


# --- Performance difference -------------------------------------------------------------------------------------------

classifiers_max_accuracy = np.max(classifiers_accuracies)
classifiers_max_f1_score = np.max(classifiers_f1_scores)
classifiers_max_mean_confidence = np.max(classifiers_mean_confidences)

difference_accuracies = np.array(combiners_accuracies) - classifiers_max_accuracy
difference_f1_scores = np.array(combiners_f1_scores) - classifiers_max_f1_score
difference_mean_confidences = np.array(combiners_mean_confidences) - classifiers_max_mean_confidence

bar1 = np.around(difference_accuracies, 3)
bar2 = np.around(difference_f1_scores, 3)
bar3 = np.around(difference_mean_confidences, 3)

barWidth = 0.2
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

plt.figure(figsize=(12, 5.5))
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)

rect1 = plt.bar(r1, bar1, color='#2b3854', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#a87f52', width=barWidth, edgecolor='white', label="F1-Score")
rect3 = plt.bar(r3, bar3, color='#52a859', width=barWidth, edgecolor='white', label="Mittlere Konfidenz")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth for r in range(len(bar1))], [comb.SHORT_NAME for comb in eval_combiner.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung (Differenz)', fontweight='bold', labelpad=15)
plt.yticks(extend_y_ticks(plt))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "102_combiner_score_differences_grouped", eval_id)
plt.close()


# === Combiner runtimes ================================================================================================

runtime_matrix = np.nan_to_num(eval_combiner.get_runtime_matrix())
combiners_train_runtimes = np.around(runtime_matrix[:, 0], 4)
combiners_combine_runtimes = np.around(runtime_matrix[:, 1], 4)
combiners_names = [c.SHORT_NAME for c in eval_combiner.get_instances()]

# --- Train runtimes ---------------------------------------------------------------------------------------------------
non_zero_indexes = np.nonzero(combiners_train_runtimes)[0]
combiners_train_non_zero_runtimes = combiners_train_runtimes[non_zero_indexes]
combiners_non_zero_names = [combiners_names[i] for i in non_zero_indexes]

# remove outliers
max_index = np.argmax(combiners_train_non_zero_runtimes)
combiners_train_non_zero_runtimes = np.delete(combiners_train_non_zero_runtimes, max_index)
del combiners_non_zero_names[max_index]

df = pd.DataFrame({'combiners_non_zero_names': combiners_non_zero_names,
                   'combiners_train_non_zero_runtimes': combiners_train_non_zero_runtimes})
df_sorted = df.sort_values('combiners_train_non_zero_runtimes')

plt.figure()
bar1 = plt.bar('combiners_non_zero_names', 'combiners_train_non_zero_runtimes', data=df_sorted, color='#93c6ed',
               width=.75)
# plt.title("Trainingslaufzeit der Fusionsmethoden")
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z91z_train_runtime_comparison", eval_id)
plt.close()

# --- Combine runtimes -------------------------------------------------------------------------------------------------

df = pd.DataFrame({'combiners_names': combiners_names, 'combiners_combine_runtimes': combiners_combine_runtimes})
df_sorted = df.sort_values('combiners_combine_runtimes')

plt.figure(figsize=(8, 4.8))
bar1 = plt.bar('combiners_names', 'combiners_combine_runtimes', data=df_sorted, color='#006aba', width=.75)
# plt.title("Fusionslaufzeit der Fusionsmethoden")
plt.xlabel("Fusionsmethoden", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z92_combine_runtime_comparison", eval_id)
plt.close()


save_evaluator(__file__, eval_id)
print("Evaluation", eval_id, "done.")
