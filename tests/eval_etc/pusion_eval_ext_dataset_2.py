import time

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation
from pusion.evaluation.evaluation_metrics import *
from pusion.input_output.file_input_output import *
from pusion.util.generator import split_into_train_and_validation_data
from pusion.util.transformer import *

# warnings.filterwarnings('error')  # halt on warning

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


eval_id = time.strftime("%Y%m%d-%H%M%S")

random_state = 1


data = np.loadtxt("/int/datasets/sdd/Sensorless_drive_diagnosis.txt", delimiter=' ')

X = data[:, :-1]
y = transform_label_vector_to_class_assignment_matrix(data[:, -1] - 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)


classifiers = [
    MLPClassifier(random_state=1),
    MLPClassifier(random_state=2),
    MLPClassifier(random_state=3),
    # KNeighborsClassifier(3),
    # KNeighborsClassifier(5),
    # KNeighborsClassifier(7),
]

for i, classifier in enumerate(classifiers):
    print("Training classifier", i, "...")
    classifier.fit(X_train, y_train)


decision_outputs = np.full((len(classifiers), y_test.shape[0], y_test.shape[1]), np.nan)
for i, classifier in enumerate(classifiers):
    print("Predicting with classifier", i, "...")
    decision_outputs[i] = classifier.predict(X_test)

print()

coverage = []
true_assignments = y_test
cr = False

np.random.seed(random_state)

y_ensemble_test, y_test, y_ensemble_valid, y_valid = \
    split_into_train_and_validation_data(decision_outputs, true_assignments, validation_size=.5)

y_ensemble_test = multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test)
y_ensemble_valid = multiclass_prediction_tensor_to_decision_tensor(y_ensemble_valid)


eval_metrics = [
    p.PerformanceMetric.ACCURACY,
    p.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE,
    p.PerformanceMetric.MICRO_JACCARD_SCORE,
    p.PerformanceMetric.MACRO_F1_SCORE,
]


print("============= Ensemble ===============")
eval_classifiers = Evaluation(*eval_metrics)
if cr:
    eval_classifiers.set_instances(['Ensemble'])
    eval_classifiers.evaluate_cr_decision_outputs(
        y_test, multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test), coverage)
else:
    instances = [('NN ' + str(i + 1)) for i in range(len(y_ensemble_test))]
    eval_classifiers.set_instances(instances)
    eval_classifiers.evaluate(y_test, multiclass_prediction_tensor_to_decision_tensor(y_ensemble_test))

print(eval_classifiers.get_report())

# ---- Mean confidence on continuous ensemble outputs
# eval_classifiers_confidence = Evaluation(p.PerformanceMetric.MEAN_CONFIDENCE)
# if cr:
#     eval_classifiers_confidence.set_instances(['Ensemble'])
#     eval_classifiers_confidence.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
# else:
#     eval_classifiers_confidence.set_instances([('ResNet ' + str(i + 1)) for i in range(len(y_ensemble_test))])
#     eval_classifiers_confidence.evaluate(y_test, y_ensemble_test)
#
# print(eval_classifiers_confidence.get_report())


# ---- GenericCombiner -------------------------------------------------------------------------------------------------
dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
dp.set_parallel(False)
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

# ---- Mean confidence on continuous combiner outputs
# eval_combiner_confidence = Evaluation(p.PerformanceMetric.MEAN_CONFIDENCE)
# eval_combiner_confidence.set_instances(dp.get_combiners())
# multi_comb_continuous_outputs = dp.get_multi_combiner_decision_output()
# if cr:
#     eval_combiner_confidence.evaluate_cr_multi_combiner_decision_outputs(y_test, multi_comb_continuous_outputs)
# else:
#     eval_combiner_confidence.evaluate(y_test, multi_comb_continuous_outputs)
# print(eval_combiner_confidence.get_report())


# # ---- ROC curves for classifiers
# for i, do in enumerate(y_ensemble_test):
#     skplt.metrics.plot_roc_curve(multiclass_assignments_to_labels(y_test), do)
#     save(plt, "000_classifier_" + str(i) + "_roc_curve", eval_id + "/roc")
#
# # ---- ROC curves for combiners with continuous outputs
# for do, comb in zip(multi_comb_continuous_outputs, dp.get_combiners()):
#     if determine_assignment_type(do) == p.AssignmentType.CONTINUOUS:
#         # skplt.metrics.plot_roc_curve(multiclass_assignments_to_labels(y_test), do)
#         # plt.title(comb.SHORT_NAME)
#         # save(plt, "001_combiner_" + comb.SHORT_NAME + "_roc_curve", eval_id + "/roc")
#         print("CONT. OUT: ", comb.SHORT_NAME)

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
classifiers_balanced_multiclass_accuracy_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE)]
classifiers_micro_jaccard_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_JACCARD_SCORE)]
classifiers_macro_f1_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(
    p.PerformanceMetric.MACRO_F1_SCORE)]

bar1 = np.around(classifiers_accuracies, 3)
bar2 = np.around(classifiers_balanced_multiclass_accuracy_scores, 3)
bar3 = np.around(classifiers_micro_jaccard_scores, 3)
bar4 = np.around(classifiers_macro_f1_scores, 3)

barWidth = 0.13
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure()
rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balancierte Trefferquote")
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
combiners_balanced_multiclass_accuracy_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE)]
combiners_micro_jaccard_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MICRO_JACCARD_SCORE)]
combiners_macro_f1_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(
    p.PerformanceMetric.MACRO_F1_SCORE)]

bar1 = np.around(combiners_accuracies, 3)
bar2 = np.around(combiners_balanced_multiclass_accuracy_scores, 3)
bar3 = np.around(combiners_micro_jaccard_scores, 3)
bar4 = np.around(combiners_macro_f1_scores, 3)

barWidth = 0.16
r1 = np.arange(len(bar1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.figure(figsize=(12, 5.5))
rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balancierte Trefferquote")
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
classifiers_max_balanced_multiclass_accuracy_score = np.max(classifiers_balanced_multiclass_accuracy_scores)
classifiers_max_micro_jaccard_score = np.max(classifiers_micro_jaccard_scores)
classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

difference_accuracies = np.array(combiners_accuracies) - classifiers_max_accuracy
difference_balanced_multiclass_accuracy_scores = np.array(combiners_balanced_multiclass_accuracy_scores) - classifiers_max_balanced_multiclass_accuracy_score
difference_micro_jaccard_scores = np.array(combiners_micro_jaccard_scores) - classifiers_max_micro_jaccard_score
difference_macro_f1_scores = np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score

bar1 = np.around(difference_accuracies, 3)
bar2 = np.around(difference_balanced_multiclass_accuracy_scores, 3)
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
rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balancierte Trefferquote")
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
classifiers_max_balanced_multiclass_accuracy_score = np.max(classifiers_balanced_multiclass_accuracy_scores)
classifiers_max_micro_jaccard_score = np.max(classifiers_micro_jaccard_scores)
classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

difference_accuracies = (np.array(combiners_accuracies) - classifiers_max_accuracy).clip(min=0)
difference_balanced_multiclass_accuracy_scores = (np.array(combiners_balanced_multiclass_accuracy_scores) -
                                   classifiers_max_balanced_multiclass_accuracy_score).clip(min=0)
difference_micro_jaccard_scores = (np.array(combiners_micro_jaccard_scores) - classifiers_max_micro_jaccard_score).clip(min=0)
difference_macro_f1_scores = (np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score).clip(min=0)

combiners = list(eval_combiner.get_instances())

for i, perf in reversed(list(enumerate(difference_accuracies))):
    if difference_accuracies[i] == difference_balanced_multiclass_accuracy_scores[i] == difference_micro_jaccard_scores[i] == \
            difference_macro_f1_scores[i] == 0:
        difference_accuracies = np.delete(difference_accuracies, i)
        difference_balanced_multiclass_accuracy_scores = np.delete(difference_balanced_multiclass_accuracy_scores, i)
        difference_micro_jaccard_scores = np.delete(difference_micro_jaccard_scores, i)
        difference_macro_f1_scores = np.delete(difference_macro_f1_scores, i)
        del combiners[i]

if len(combiners) > 0:
    bar1 = np.around(difference_accuracies, 3)
    bar2 = np.around(difference_balanced_multiclass_accuracy_scores, 3)
    bar3 = np.around(difference_micro_jaccard_scores, 3)
    bar4 = np.around(difference_macro_f1_scores, 3)

    barWidth = 0.16
    r1 = np.arange(len(bar1))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    plt.figure(figsize=(12, 5.5))
    rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
    rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balancierte Trefferquote")
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

# labels=np.arange(4)
cm = confusion_matrix(multiclass_assignments_to_labels(y_test), multiclass_assignments_to_labels(y_test))
display = ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
display.plot(cmap='binary')
# plt.title("Ground Truth")
plt.xlabel('vorhergesagte Klassen', fontweight='bold', labelpad=15)
plt.ylabel('wahre Klassen', fontweight='bold', labelpad=15)
save(plt, "000_ground_truth_confusion_matrix", eval_id + "/cm")
plt.close()

if not cr:
    for i, dt in enumerate(y_ensemble_test):
        # labels=np.arange(4)
        cm = confusion_matrix(multiclass_assignments_to_labels(y_test), multiclass_assignments_to_labels(dt))
        display = ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
        display.plot(cmap='binary')
        # plt.title("Classifier " + str(i))
        plt.xlabel('vorhergesagte Klassen', fontweight='bold', labelpad=15)
        plt.ylabel('wahre Klassen', fontweight='bold', labelpad=15)
        save(plt, "001_classifier_" + str(i) + "_confusion_matrix", eval_id + "/cm")
        plt.close()

if not cr:
    for i, comb in enumerate(eval_combiner.get_instances()):
        cm = confusion_matrix(multiclass_assignments_to_labels(y_test),
                              multiclass_assignments_to_labels(multi_comb_decision_outputs[i]))  # labels=np.arange(4)
        display = ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
        display.plot(cmap='binary')
        # plt.title(comb.SHORT_NAME)
        plt.xlabel('vorhergesagte Klassen', fontweight='bold', labelpad=15)
        plt.ylabel('wahre Klassen', fontweight='bold', labelpad=15)
        save(plt, "002_" + str(i) + "_" + comb.SHORT_NAME + "_combiner_confusion_matrix", eval_id + "/cm")
        plt.close()

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
