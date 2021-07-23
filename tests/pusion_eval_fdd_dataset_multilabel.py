import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *
from pusion.util.transformer import *

warnings.filterwarnings('error')  # halt on warning

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


eval_id = time.strftime("%Y%m%d-%H%M%S")

random_state = 1

dataset_files = [
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_01.pickle',
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_02.pickle',  # (e1)
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_03.pickle',  # (e1)
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_04.pickle',  # (e1)
    #
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_05.pickle',
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_06.pickle',
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_07.pickle',

    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_08.pickle',
    # '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_09.pickle',
    '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_10.pickle',
    '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_11.pickle',
    '/int/DFF_DL_data/models_for_fusion/Time-SE-ResNet_DF3_12.pickle',
]

data = load_pickle_files_as_data(dataset_files)

y_ensemble_valid = [np.roll(data[i]['Y_test_predictions'], shift=1, axis=1) for i in range(len(dataset_files))]
y_ensemble_valid = tensorize(y_ensemble_valid)
y_valid = np.roll(data[0]['Y_test'], shift=1, axis=1)

y_ensemble_test = [np.roll(data[i]['Y_test_for_fusion_predictions'], shift=1, axis=1) for i in range(len(dataset_files))]
y_ensemble_test = tensorize(y_ensemble_test)
y_test = np.roll(data[0]['Y_test_for_fusion'], shift=1, axis=1)

coverage = []
cr = False

np.random.seed(random_state)

y_ensemble_test = multilabel_prediction_tensor_to_decision_tensor(y_ensemble_test)
y_ensemble_valid = multilabel_prediction_tensor_to_decision_tensor(y_ensemble_valid)


eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.MICRO_F1_SCORE,
    p.PerformanceMetric.MICRO_JACCARD_SCORE,
    p.PerformanceMetric.MACRO_F1_SCORE,
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
if cr:
    eval_classifiers.set_instances(['Ensemble'])
    eval_classifiers.evaluate_cr_decision_outputs(
        y_test, multilabel_prediction_tensor_to_decision_tensor(y_ensemble_test), coverage)
else:
    eval_classifiers.set_instances([('ResNet ' + str(i + 1)) for i in range(len(y_ensemble_test))])
    eval_classifiers.evaluate(y_test, multilabel_prediction_tensor_to_decision_tensor(y_ensemble_test))

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
multi_comb_decision_outputs = multilabel_prediction_tensor_to_decision_tensor(dp.get_multi_combiner_decision_output())
if cr:
    eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, multi_comb_decision_outputs)
else:
    eval_combiner.evaluate(y_test, multi_comb_decision_outputs)
print(eval_combiner.get_report())

# === Plots ============================================================================================================
meanprops = dict(markerfacecolor='black', markeredgecolor='white')
plt.rc('axes', titlesize=12)     # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels


def extend_y_ticks(plot):
    y_ticks = plot.yticks()[0].tolist()
    step = y_ticks[-1] - y_ticks[-2]
    y_ticks.insert(0, y_ticks[0] - step)
    y_ticks.append(y_ticks[-1] + step)
    return y_ticks


def extend_y_ticks_upper_bound(plot):
    y_ticks = plot.yticks()[0].tolist()
    step = y_ticks[-1] - y_ticks[-2]
    y_ticks.append(y_ticks[-1] + step)
    return y_ticks


def extend_x_ticks_upper_bound(plot):
    x_ticks = plot.xticks()[0].tolist()
    step = x_ticks[-1] - x_ticks[-2]
    x_ticks.append(x_ticks[-1] + step)
    return x_ticks


# --- Ensemble performance ---------------------------------------------------------------------------------------------
classifiers_accuracies = [t[1] for t in eval_classifiers.get_instance_performance_tuples(p.PerformanceMetric.ACCURACY)]
classifiers_micro_f1_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_F1_SCORE)]
classifiers_micro_jaccard_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_JACCARD_SCORE)]
classifiers_macro_f1_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MACRO_F1_SCORE)]

bar1 = np.around(classifiers_accuracies, 3)
bar2 = np.around(classifiers_micro_f1_scores, 3)
bar3 = np.around(classifiers_micro_jaccard_scores, 3)
bar4 = np.around(classifiers_macro_f1_scores, 3)

barWidth = 0.13
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure()
rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#346899', width=barWidth, edgecolor='white', label="Micro F1-Score")
rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro Jaccard-Score")
rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

plt.xlabel('Ensemble', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))],
           [str(instance) for instance in eval_classifiers.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung', fontweight='bold', labelpad=15)
plt.yticks(np.arange(0, 1.1, .1))
plt.ylim((0, 1.2))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)
plt.bar_label(rect4, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "100_classifier_scores_grouped", eval_id)
plt.close()


# --- Combiners performance --------------------------------------------------------------------------------------------
combiners_accuracies = [t[1] for t in eval_combiner.get_instance_performance_tuples(p.PerformanceMetric.ACCURACY)]
combiners_micro_f1_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_F1_SCORE)]
combiners_micro_jaccard_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_JACCARD_SCORE)]
combiners_macro_f1_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MACRO_F1_SCORE)]

bar1 = np.around(combiners_accuracies, 3)
bar2 = np.around(combiners_micro_f1_scores, 3)
bar3 = np.around(combiners_micro_jaccard_scores, 3)
bar4 = np.around(combiners_macro_f1_scores, 3)

barWidth = 0.16
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 5.5))
rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#346899', width=barWidth, edgecolor='white', label="Micro F1-Score")
rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro Jaccard-Score")
rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [comb.SHORT_NAME for comb in eval_combiner.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung', fontweight='bold', labelpad=15)
plt.yticks(np.arange(0, 1.1, .1))
plt.ylim((0, 1.2))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)
plt.bar_label(rect4, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "101_combiner_scores_grouped", eval_id)
plt.close()


# --- Performance difference -------------------------------------------------------------------------------------------

classifiers_max_accuracy = np.max(classifiers_accuracies)
classifiers_max_micro_f1_score = np.max(classifiers_micro_f1_scores)
classifiers_max_micro_jaccard_score = np.max(classifiers_micro_jaccard_scores)
classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

difference_accuracies = np.array(combiners_accuracies) - classifiers_max_accuracy
difference_micro_f1_scores = np.array(combiners_micro_f1_scores) - classifiers_max_micro_f1_score
difference_micro_jaccard_scores = np.array(combiners_micro_jaccard_scores) - classifiers_max_micro_jaccard_score
difference_macro_f1_scores = np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score

bar1 = np.around(difference_accuracies, 3)
bar2 = np.around(difference_micro_f1_scores, 3)
bar3 = np.around(difference_micro_jaccard_scores, 3)
bar4 = np.around(difference_macro_f1_scores, 3)

barWidth = 0.16
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 5.5))
plt.axhline(y=0, color='gray', linestyle='-', linewidth=1)

rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#346899', width=barWidth, edgecolor='white', label="Micro F1-Score")
rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro Jaccard-Score")
rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [comb.SHORT_NAME for comb in eval_combiner.get_instances()])
plt.xlim(-.5, np.max(r1) + 1.5)
plt.ylabel('Wertung (Differenz)', fontweight='bold', labelpad=15)
plt.yticks(extend_y_ticks(plt))

plt.bar_label(rect1, padding=3, rotation=90)
plt.bar_label(rect2, padding=3, rotation=90)
plt.bar_label(rect3, padding=3, rotation=90)
plt.bar_label(rect4, padding=3, rotation=90)

plt.legend(loc="lower right")
plt.tight_layout()
save(plt, "102_combiner_score_differences_grouped", eval_id)
plt.close()


# --- Performance improvement ------------------------------------------------------------------------------------------

classifiers_max_accuracy = np.max(classifiers_accuracies)
classifiers_max_micro_f1_score = np.max(classifiers_micro_f1_scores)
classifiers_max_micro_jaccard_score = np.max(classifiers_micro_jaccard_scores)
classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

difference_accuracies = (np.array(combiners_accuracies) - classifiers_max_accuracy).clip(min=0)
difference_micro_f1_scores = (np.array(combiners_micro_f1_scores) -
                                   classifiers_max_micro_f1_score).clip(min=0)
difference_micro_jaccard_scores = (np.array(combiners_micro_jaccard_scores) - classifiers_max_micro_jaccard_score).clip(min=0)
difference_macro_f1_scores = (np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score).clip(min=0)

combiners = list(eval_combiner.get_instances())

for i, perf in reversed(list(enumerate(difference_accuracies))):
    if difference_accuracies[i] == difference_micro_f1_scores[i] == difference_micro_jaccard_scores[i] == \
            difference_macro_f1_scores[i] == 0:
        difference_accuracies = np.delete(difference_accuracies, i)
        difference_micro_f1_scores = np.delete(difference_micro_f1_scores, i)
        difference_micro_jaccard_scores = np.delete(difference_micro_jaccard_scores, i)
        difference_macro_f1_scores = np.delete(difference_macro_f1_scores, i)
        del combiners[i]

if len(combiners) > 0:
    bar1 = np.around(difference_accuracies, 3)
    bar2 = np.around(difference_micro_f1_scores, 3)
    bar3 = np.around(difference_micro_jaccard_scores, 3)
    bar4 = np.around(difference_macro_f1_scores, 3)

    barWidth = 0.16
    r1 = np.arange(len(bar1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    plt.figure(figsize=(12, 5.5))
    rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
    rect2 = plt.bar(r2, bar2, color='#346899', width=barWidth, edgecolor='white', label="Micro F1-Score")
    rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro Jaccard-Score")
    rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

    plt.xlabel('Fusionsmethoden', fontweight='bold', labelpad=15)
    plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [comb.SHORT_NAME for comb in combiners])
    plt.xlim(-.5, np.max(r1) + 2)
    plt.ylabel('Wertung (Differenz)', fontweight='bold', labelpad=15)
    plt.yticks(extend_y_ticks_upper_bound(plt))

    plt.bar_label(rect1, padding=3, rotation=90)
    plt.bar_label(rect2, padding=3, rotation=90)
    plt.bar_label(rect3, padding=3, rotation=90)
    plt.bar_label(rect4, padding=3, rotation=90)

    plt.legend(loc="lower right")
    plt.tight_layout()
    save(plt, "103_combiner_score_positive_improvement_grouped", eval_id)
    plt.close()


# === Confusion matrices ===============================================================================================

plt.rc('axes', titlesize=27)
plt.rc('axes', labelsize=22)
matplotlib.rcParams.update({'font.size': 24})

for i, dt in enumerate(y_ensemble_test):
    fig, axes = plt.subplots(1, y_test.shape[1], figsize=(25, 3))
    for j in range(y_test.shape[1]):
        cm = confusion_matrix(y_test[:, j], dt[:, j])
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        display.plot(ax=axes[j], cmap='binary')
        display.ax_.set_title(f'$\\omega_{j}$', pad=10)
        display.ax_.set_xlabel('')
        display.ax_.set_ylabel('')
        display.im_.colorbar.remove()
    plt.tight_layout()
    save(plt, "001_classifier_" + str(i) + "_confusion_matrix", eval_id + "/cm")
    plt.close()

for i, comb in enumerate(eval_combiner.get_instances()):
    fig, axes = plt.subplots(1, y_test.shape[1], figsize=(25, 3))
    for j in range(y_test.shape[1]):
        cm = confusion_matrix(y_test[:, j], multi_comb_decision_outputs[i][:, j])
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        display.plot(ax=axes[j], cmap='binary')
        display.ax_.set_title(f'$\\omega_{j}$', pad=10)
        display.ax_.set_xlabel('')
        display.ax_.set_ylabel('')
        display.im_.colorbar.remove()
    plt.tight_layout()
    save(plt, "002_" + str(i) + "_" + comb.SHORT_NAME + "_combiner_confusion_matrix", eval_id + "/cm")
    plt.close()


matplotlib.rcParams.update(matplotlib.rcParamsDefault)


# === Combiner runtimes ================================================================================================
plt.rcParams.update({'font.size': 13})

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
plt.xlabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z90_train_runtime_comparison", eval_id)
plt.close()

# --- horizontal bar plot
plt.figure()
bar1 = plt.barh('combiners_non_zero_names', 'combiners_train_non_zero_runtimes', data=df_sorted, color='#b5670e',
                height=.2)
plt.xlabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z91_train_runtime_comparison_h", eval_id)
plt.close()

# --- Combine runtimes -------------------------------------------------------------------------------------------------

df = pd.DataFrame({'combiners_names': combiners_names, 'combiners_combine_runtimes': combiners_combine_runtimes})
df_sorted = df.sort_values('combiners_combine_runtimes')

plt.figure(figsize=(8, 4.8))
bar1 = plt.bar('combiners_names', 'combiners_combine_runtimes', data=df_sorted, color='#006aba', width=.75)
plt.xlabel("Fusionsmethoden", fontweight='bold', labelpad=15)
plt.ylabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z92_combine_runtime_comparison", eval_id)
plt.close()

# --- horizontal bar plot
plt.figure()
bar1 = plt.barh('combiners_names', 'combiners_combine_runtimes', data=df_sorted, color='#915006', height=.4)
plt.xlabel("Laufzeit (s)", fontweight='bold', labelpad=15)
plt.ylabel("Fusionsmethode", fontweight='bold', labelpad=15)
plt.xticks(extend_x_ticks_upper_bound(plt))
plt.bar_label(bar1, padding=3)
plt.tight_layout()
save(plt, "z93_combine_runtime_comparison_h", eval_id)
plt.close()

save_evaluator(__file__, eval_id)
print("Evaluation", eval_id, "done.")
