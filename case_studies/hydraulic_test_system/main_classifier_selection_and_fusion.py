import os
import re

import matplotlib.pyplot as plt
import pandas as pd

import pusion
from pusion.evaluation.evaluation import Evaluation
from pusion.input_output.file_input_output import *
from pusion.util.transformer import *
from pusion.evaluation.evaluation_metrics import *

# wider console output
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def main(paths):

    # dict containing the evaluation results of all iterations of the k + 1 fold CV
    overall_k_1_fold_cv_results = {}

    # scalar_eval_metrics return scalar values as a performance measure
    scalar_eval_metrics = [
        pusion.PerformanceMetric.ACCURACY,
        pusion.PerformanceMetric.ERROR_RATE,
        pusion.PerformanceMetric.WEIGHTED_F1_SCORE,
        pusion.PerformanceMetric.MULTICLASS_FDR,
        pusion.PerformanceMetric.FALSE_ALARM_RATE,
        pusion.PerformanceMetric.MULTICLASS_BRIER_SCORE,
        pusion.PerformanceMetric.MULTICLASS_WEIGHTED_SCIKIT_AUC_ROC_SCORE
    ]

    # scalar_eval_metrics return vectors (class-wise values) as a performance measure
    vectorial_eval_metrics = [
        pusion.PerformanceMetric.MULTICLASS_CLASS_WISE_PRECISION,
        pusion.PerformanceMetric.MULTICLASS_CLASS_WISE_RECALL,
        pusion.PerformanceMetric.MULTICLASS_CLASS_WISE_AVG_PRECISION,
        pusion.PerformanceMetric.MULTICLASS_PYTORCH_AUC_ROC,
        pusion.PerformanceMetric.MULTICLASS_WEIGHTED_PYTORCH_AUC_ROC,
        pusion.PerformanceMetric.MULTICLASS_AUC_PRECISION_RECALL_CURVE
    ]

    # analytical_eval_metrics return data (e.g. matrices) for histograms and curves
    # warning: no numpy access (result dicts need to be traversed manually).
    analytical_eval_metrics = [
        # pusion.PerformanceMetric.MULTICLASS_AUC_PRECISION_RECALL_CURVE
    ]

    selected_classifiers = [3, 8, 10, 12, 14]

    for i_iter, path in enumerate(paths):
        with open(path, 'rb') as handle:
            cv_results = pickle.load(handle)
        cv_runs = [irun for irun in cv_results.keys() if re.match('^train\\d+_val\\d+$', irun)]

        for i_cv_run, cv_run in enumerate(cv_runs):

            # crisp predictions
            y_ensemble_valid = cv_results[cv_run]['y_ensemble_valid'][selected_classifiers]
            y_ensemble_test = cv_results[cv_run]['y_ensemble_test'][selected_classifiers]

            # valid and test datasets
            curr_y_valid_fold = cv_results[cv_run]['j_current_val_fold']
            y_valid = cv_results['Y_folds'][curr_y_valid_fold]

            curr_y_test_fold = cv_results[cv_run]['i_current_test_fold']
            y_test = cv_results['Y_folds'][curr_y_test_fold]

            classifiers = [cv_results['classifiers'][i] for i in selected_classifiers]

            # --- calculate diversity measures
            curr_iter = "iter_" + str(i_iter) + "_cvrun_" + str(i_cv_run)
            if 'diversity_measures' not in overall_k_1_fold_cv_results:
                overall_k_1_fold_cv_results['diversity_measures'] = {}
                overall_k_1_fold_cv_results['diversity_measures']['correlation'] = {}
                overall_k_1_fold_cv_results['diversity_measures']['kappa_statistic'] = {}
                overall_k_1_fold_cv_results['diversity_measures']['double_fault'] = {}

            indices_corr, pair_corr = pairwise_correlation(y_ensemble_test, y_test, return_type='list')
            corr_dict = dict()
            corr_dict['indices'] = indices_corr
            corr_dict['correlation'] = pair_corr
            overall_k_1_fold_cv_results['diversity_measures']['correlation'][curr_iter] = corr_dict

            indices_kappa, pair_kappa = pairwise_kappa_statistic(y_ensemble_test, y_test, return_type='list')
            kappa_statistic_dict = dict()
            kappa_statistic_dict['indices'] = indices_kappa
            kappa_statistic_dict['kappa_statistic'] = pair_kappa
            overall_k_1_fold_cv_results['diversity_measures']['kappa_statistic'][curr_iter] = kappa_statistic_dict

            indices_doublef, pair_doublef = pairwise_double_fault(y_ensemble_test, y_test, return_type='list')
            doublef_dict = dict()
            doublef_dict['indices'] = indices_doublef
            doublef_dict['double_fault'] = pair_doublef
            overall_k_1_fold_cv_results['diversity_measures']['double_fault'][curr_iter] = pair_doublef

            # ---- Evaluate classifiers --------------------------------------------------------------------------------
            eval_classifiers = pusion.Evaluation(*scalar_eval_metrics)
            eval_classifiers.set_instances(classifiers)
            eval_classifiers.evaluate(y_test, y_ensemble_test)
            print(eval_classifiers.get_report())

            # ---- Evaluate GenericCombiner ----------------------------------------------------------------------------
            dp = pusion.DecisionProcessor(pusion.Configuration(method=pusion.Method.GENERIC))
            dp.set_parallel(False)
            dp.train(y_ensemble_valid, y_valid)
            dp.combine(y_ensemble_test)

            eval_combiner = Evaluation(*scalar_eval_metrics)
            eval_combiner.set_instances(dp.get_combiners())
            eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
            # use multiclass_prediction_tensor_to_decision_tensor to get crisp class assignments.
            multi_comb_decision_outputs = multiclass_prediction_tensor_to_decision_tensor(dp.get_multi_combiner_decision_output())
            eval_combiner.evaluate(y_test, multi_comb_decision_outputs)

            dp.set_evaluation(eval_combiner)
            print(dp.report())

            combiners = dp.get_combiners()


            # ---- scalar metrics --------------------------------------------------------------------------------------
            if 'cv_scalar_perf_tensor' not in overall_k_1_fold_cv_results:
                overall_k_1_fold_cv_results['meta_data'] = {}
                overall_k_1_fold_cv_results['meta_data']['cv_iterations'] = paths
                overall_k_1_fold_cv_results['meta_data']['cv_runs'] = cv_runs
                overall_k_1_fold_cv_results['meta_data']['classifier_instances'] = classifiers
                overall_k_1_fold_cv_results['meta_data']['combiner_instances'] = combiners
                overall_k_1_fold_cv_results['meta_data']['scalar_eval_metrics'] = scalar_eval_metrics
                overall_k_1_fold_cv_results['meta_data']['vectorial_metrics'] = vectorial_eval_metrics

                overall_k_1_fold_cv_results['cv_scalar_perf_tensor'] = {}
                overall_k_1_fold_cv_results['cv_scalar_perf_tensor']['axes'] = {
                    0: 'i_iter',
                    1: 'i_cv_run',
                    2: 'i_instance',
                    3: 'i_scalar_eval_metric'
                }
                overall_k_1_fold_cv_results['cv_scalar_perf_tensor']['ensemble'] = \
                    np.full((len(paths), len(cv_runs), len(classifiers), len(scalar_eval_metrics)), np.nan)
                overall_k_1_fold_cv_results['cv_scalar_perf_tensor']['combiners'] = \
                    np.full((len(paths), len(cv_runs), len(combiners), len(scalar_eval_metrics)), np.nan)

            # classifiers
            overall_k_1_fold_cv_results['cv_scalar_perf_tensor']['ensemble'][i_iter, i_cv_run, :, :] = eval_classifiers.get_performance_matrix()

            # combiners
            overall_k_1_fold_cv_results['cv_scalar_perf_tensor']['combiners'][i_iter, i_cv_run, :, :] = \
                eval_combiner.get_performance_matrix()

            # ---- vectorial metrics -----------------------------------------------------------------------------------
            classes = np.arange(y_test.shape[1])

            if 'cv_vectorial_perf_tensor' not in overall_k_1_fold_cv_results:
                overall_k_1_fold_cv_results['cv_vectorial_perf_tensor'] = {}
                overall_k_1_fold_cv_results['cv_vectorial_perf_tensor']['axes'] = {
                    0: 'i_iter',
                    1: 'i_cv_run',
                    2: 'i_instance',
                    3: 'i_vectorial_eval_metric',
                    4: 'i_class'
                }
                overall_k_1_fold_cv_results['cv_vectorial_perf_tensor']['ensemble'] = \
                    np.full((len(paths), len(cv_runs), len(classifiers), len(vectorial_eval_metrics), len(classes)),
                            np.nan)
                overall_k_1_fold_cv_results['cv_vectorial_perf_tensor']['combiners'] = \
                    np.full((len(paths), len(cv_runs), len(combiners), len(vectorial_eval_metrics), len(classes)), np.nan)


            # classifiers
            for i_vem, vem in enumerate(vectorial_eval_metrics):
                for i_clf, clf in enumerate(classifiers):
                    overall_k_1_fold_cv_results['cv_vectorial_perf_tensor']['ensemble'][i_iter, i_cv_run, i_clf, i_vem, :] = \
                        vem(y_test, y_ensemble_test[i_clf])

            # combiners
            for i_vem, vem in enumerate(vectorial_eval_metrics):
                for i_comb, comb in enumerate(combiners):
                    overall_k_1_fold_cv_results['cv_vectorial_perf_tensor']['combiners'][i_iter, i_cv_run, i_comb, i_vem, :] = \
                        vem(y_test, multi_comb_decision_outputs[i_comb])

            # ---- analytical metrics ----------------------------------------------------------------------------------
            if 'cv_analytical_metrics' not in overall_k_1_fold_cv_results:
                overall_k_1_fold_cv_results['cv_analytical_metrics'] = {}
                overall_k_1_fold_cv_results['cv_analytical_metrics']['ensemble'] = {}
                overall_k_1_fold_cv_results['cv_analytical_metrics']['combiners'] = {}

            # classifiers
            for i_aem, aem in enumerate(analytical_eval_metrics):
                for i_clf, clf in enumerate(classifiers):
                    eval_index = 'i_iter_' + str(i_iter) + '--i_cv_run_' + str(i_cv_run) + \
                                 '--i_classifier_' + str(i_clf) + '--i_aem_' + str(i_aem)
                    overall_k_1_fold_cv_results['cv_analytical_metrics']['ensemble'][eval_index] = {
                        'i_iter': i_iter,
                        'i_cv_run': i_cv_run,
                        'i_classifier': i_clf,
                        'i_aem': i_aem,
                        'result': aem(y_test, multi_comb_decision_outputs[i_clf])
                    }

            # combiners
            for i_aem, aem in enumerate(analytical_eval_metrics):
                for i_comb, comb in enumerate(combiners):
                    eval_index = 'i_iter_' + str(i_iter) + '--i_cv_run_' + str(i_cv_run) + \
                                 '--i_combiner_' + str(i_comb) + '--i_aem_' + str(i_aem)
                    overall_k_1_fold_cv_results['cv_analytical_metrics']['combiners'][eval_index] = {
                        'i_iter': i_iter,
                        'i_cv_run': i_cv_run,
                        'i_combiner': i_comb,
                        'i_aem': i_aem,
                        'result': aem(y_test, multi_comb_decision_outputs[i_comb])
                    }

            continue

            # === Plots ================================================================================================
            meanprops = dict(markerfacecolor='black', markeredgecolor='white')
            plt.rc('axes', titlesize=12)  # fontsize of the axes title
            plt.rc('axes', labelsize=12)  # fontsize of the x and y labels

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

            # --- Ensemble performance ---------------------------------------------------------------------------------
            classifier_names = [str(i) for i in eval_classifiers.get_instances()]
            classifiers_accuracies = [t[1] for t in eval_classifiers.get_instance_performance_tuples(pusion.PerformanceMetric.ACCURACY)]
            classifiers_balanced_multiclass_accuracy_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(pusion.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE)]
            classifiers_micro_recall = [t[1] for t in eval_classifiers.get_instance_performance_tuples(pusion.PerformanceMetric.MICRO_RECALL)]
            classifiers_macro_f1_scores = [t[1] for t in eval_classifiers.get_instance_performance_tuples(pusion.PerformanceMetric.MACRO_F1_SCORE)]
            classifiers_error_rates = [t[1] for t in eval_classifiers.get_instance_performance_tuples(pusion.PerformanceMetric.ERROR_RATE)]

            curr_performance_dict = dict()
            curr_performance_dict['classifier_names'] = classifier_names
            curr_performance_dict['classifiers_accuracies'] = classifiers_accuracies
            curr_performance_dict['_classifiers_error_rates'] = classifiers_error_rates

            bar1 = np.around(classifiers_accuracies, 3)
            bar2 = np.around(classifiers_balanced_multiclass_accuracy_scores, 3)
            bar3 = np.around(classifiers_micro_recall, 3)
            bar4 = np.around(classifiers_macro_f1_scores, 3)

            barWidth = 0.13
            r1 = np.arange(len(bar1))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            r4 = [x + barWidth for x in r3]

            plt.figure(figsize=(12, 8))
            rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
            rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balancierte Trefferquote")
            rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro F1-Score")
            rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

            plt.xlabel('Ensemble', fontweight='bold', labelpad=15)
            plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [str(instance) for instance in eval_classifiers.get_instances()], rotation=45, ha='right', rotation_mode='anchor')
            plt.xlim(-.5, np.max(r1) + 1.5)
            plt.ylabel('Score', fontweight='bold', labelpad=15)
            plt.yticks(np.arange(0, 1.1, .1))
            plt.ylim((0, 1.2))

            plt.bar_label(rect1, padding=3, rotation=90)
            plt.bar_label(rect2, padding=3, rotation=90)
            plt.bar_label(rect3, padding=3, rotation=90)
            plt.bar_label(rect4, padding=3, rotation=90)

            plt.legend(loc="lower right")
            plt.grid()
            plt.tight_layout()
            plt.savefig("./results/100_classifier_scores_grouped" + ".svg", bbox_inches="tight")
            plt.savefig("./results/100_classifier_scores_grouped" + ".pdf", bbox_inches="tight")
            plt.close()

            # --- Combiners performance --------------------------------------------------------------------------------
            combiner_names = [i.SHORT_NAME for i in combiners]
            combiners_accuracies = [t[1] for t in eval_combiner.get_instance_performance_tuples(pusion.PerformanceMetric.ACCURACY)]
            combiners_balanced_multiclass_accuracy_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(pusion.PerformanceMetric.BALANCED_MULTICLASS_ACCURACY_SCORE)]
            combiners_micro_recall = [t[1] for t in eval_combiner.get_instance_performance_tuples(pusion.PerformanceMetric.MICRO_RECALL)]
            combiners_macro_f1_scores = [t[1] for t in eval_combiner.get_instance_performance_tuples(pusion.PerformanceMetric.MACRO_F1_SCORE)]
            combiners_error_rates = [t[1] for t in eval_combiner.get_instance_performance_tuples(pusion.PerformanceMetric.ERROR_RATE)]

            bar1 = np.around(combiners_accuracies, 3)
            bar2 = np.around(combiners_balanced_multiclass_accuracy_scores, 3)
            bar3 = np.around(combiners_micro_recall, 3)
            bar4 = np.around(combiners_macro_f1_scores, 3)

            barWidth = 0.16
            r1 = np.arange(len(bar1))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            r4 = [x + barWidth for x in r3]

            plt.figure(figsize=(12, 5.5))
            rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Accuracy")
            rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white', label="Balanced accuracy")
            rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro F1-score")
            rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-score")

            plt.xlabel('Fusion methods', fontweight='bold', labelpad=15, rotation=45)
            plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [comb.SHORT_NAME for comb in eval_combiner.get_instances()])
            plt.xlim(-.5, np.max(r1) + 1.5)
            plt.ylabel('Score', fontweight='bold', labelpad=15)
            plt.yticks(np.arange(0, 1.1, .1))
            plt.ylim((0, 1.2))

            plt.bar_label(rect1, padding=3, rotation=90)
            plt.bar_label(rect2, padding=3, rotation=90)
            plt.bar_label(rect3, padding=3, rotation=90)
            plt.bar_label(rect4, padding=3, rotation=90)

            plt.legend(loc="lower right")
            plt.grid()
            plt.tight_layout()

            plt.savefig("./results/101_combiner_scores_grouped" + ".svg", bbox_inches="tight")
            plt.savefig("./results/101_combiner_scores_grouped" + ".pdf", bbox_inches="tight")
            plt.close()

            # --- Performance difference -------------------------------------------------------------------------------
            classifiers_max_accuracy = np.max(classifiers_accuracies)
            classifiers_max_balanced_multiclass_accuracy_score = np.max(classifiers_balanced_multiclass_accuracy_scores)
            classifiers_max_micro_recall_score = np.max(classifiers_micro_recall)
            classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

            difference_accuracies = np.array(combiners_accuracies) - classifiers_max_accuracy
            difference_balanced_multiclass_accuracy_scores = np.array(combiners_balanced_multiclass_accuracy_scores) - classifiers_max_balanced_multiclass_accuracy_score
            difference_micro_recall = np.array(combiners_micro_recall) - classifiers_max_micro_recall_score
            difference_macro_f1_scores = np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score

            bar1 = np.around(difference_accuracies, 3)
            bar2 = np.around(difference_balanced_multiclass_accuracy_scores, 3)
            bar3 = np.around(difference_micro_recall, 3)
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
            rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro F1-Score")
            rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

            plt.xlabel('Fusion methods', fontweight='bold', labelpad=15)
            plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))],[comb.SHORT_NAME for comb in eval_combiner.get_instances()])
            plt.xlim(-.5, np.max(r1) + 1.5)
            plt.ylabel('Score (difference)', fontweight='bold', labelpad=15)
            plt.yticks(extend_y_ticks(plt))

            plt.bar_label(rect1, padding=3, rotation=90)
            plt.bar_label(rect2, padding=3, rotation=90)
            plt.bar_label(rect3, padding=3, rotation=90)
            plt.bar_label(rect4, padding=3, rotation=90)

            plt.legend(loc="lower right")
            plt.grid()
            plt.tight_layout()
            plt.savefig("./results/102_combiner_score_differences_grouped" + ".svg", bbox_inches="tight")
            plt.savefig("./results/102_combiner_score_differences_grouped" + ".pdf", bbox_inches="tight")
            plt.close()

            # --- Performance improvement ------------------------------------------------------------------------------
            classifiers_max_accuracy = np.max(classifiers_accuracies)
            classifiers_max_balanced_multiclass_accuracy_score = np.max(classifiers_balanced_multiclass_accuracy_scores)
            classifiers_max_micro_recall_score = np.max(classifiers_micro_recall)
            classifiers_max_macro_f1_score = np.max(classifiers_macro_f1_scores)

            difference_accuracies = (np.array(combiners_accuracies) - classifiers_max_accuracy).clip(min=0)
            difference_balanced_multiclass_accuracy_scores = (np.array(combiners_balanced_multiclass_accuracy_scores) - classifiers_max_balanced_multiclass_accuracy_score).clip(min=0)
            difference_micro_recall = (np.array(combiners_micro_recall) - classifiers_max_micro_recall_score).clip(min=0)
            difference_macro_f1_scores = (np.array(combiners_macro_f1_scores) - classifiers_max_macro_f1_score).clip(min=0)

            combiners = list(eval_combiner.get_instances())

            for i, perf in reversed(list(enumerate(difference_accuracies))):
                if difference_accuracies[i] == difference_balanced_multiclass_accuracy_scores[i] == difference_micro_recall[i] == difference_macro_f1_scores[i] == 0:
                    difference_accuracies = np.delete(difference_accuracies, i)
                    difference_balanced_multiclass_accuracy_scores = np.delete(difference_balanced_multiclass_accuracy_scores, i)
                    difference_micro_recall = np.delete(difference_micro_recall, i)
                    difference_macro_f1_scores = np.delete(difference_macro_f1_scores, i)
                    del combiners[i]

            if len(combiners) > 0:
                bar1 = np.around(difference_accuracies, 3)
                bar2 = np.around(difference_balanced_multiclass_accuracy_scores, 3)
                bar3 = np.around(difference_micro_recall, 3)
                bar4 = np.around(difference_macro_f1_scores, 3)

                barWidth = 0.16
                r1 = np.arange(len(bar1))
                r2 = [x + barWidth for x in r1]
                r3 = [x + barWidth for x in r2]
                r4 = [x + barWidth for x in r3]

                plt.figure(figsize=(12, 5.5))
                rect1 = plt.bar(r1, bar1, color='#7a9fc2', width=barWidth, edgecolor='white', label="Trefferquote")
                rect2 = plt.bar(r2, bar2, color='#7d2150', width=barWidth, edgecolor='white',label="Balancierte Trefferquote")
                rect3 = plt.bar(r3, bar3, color='#b55b53', width=barWidth, edgecolor='white', label="Micro F1-Score")
                rect4 = plt.bar(r4, bar4, color='#197435', width=barWidth, edgecolor='white', label="Macro F1-Score")

                plt.xlabel('Fusion methods', fontweight='bold', labelpad=15)
                plt.xticks([r + barWidth * 1.5 for r in range(len(bar1))], [comb.SHORT_NAME for comb in combiners])
                plt.xlim(-.5, np.max(r1) + 2)
                plt.ylabel('Score (difference)', fontweight='bold', labelpad=15)
                plt.yticks(extend_y_ticks_upper_bound(plt))

                plt.bar_label(rect1, padding=3, rotation=90)
                plt.bar_label(rect2, padding=3, rotation=90)
                plt.bar_label(rect3, padding=3, rotation=90)
                plt.bar_label(rect4, padding=3, rotation=90)

                plt.legend(loc="lower right")
                plt.tight_layout()
                plt.savefig("./results/103_combiner_score_positive_improvement_grouped" + ".svg", bbox_inches="tight")
                plt.savefig("./results/103_combiner_score_positive_improvement_grouped" + ".pdf", bbox_inches="tight")
                plt.close()

            # === Confusion matrices ===================================================================================

            import sklearn as sk
            cm = sk.metrics.confusion_matrix(multiclass_assignments_to_labels(y_test), multiclass_assignments_to_labels(y_test))
            display = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
            display.plot(cmap='binary')
            plt.xlabel('predicted classes', fontweight='bold', labelpad=15)
            plt.ylabel('true classes', fontweight='bold', labelpad=15)
            plt.savefig("./results/000_ground_truth_confusion_matrix" + ".svg", bbox_inches="tight")
            plt.savefig("./results/000_ground_truth_confusion_matrix" + ".pdf", bbox_inches="tight")
            plt.close()

            for i, dt in enumerate(y_ensemble_test):
                cm = sk.metrics.confusion_matrix(multiclass_assignments_to_labels(y_test), multiclass_assignments_to_labels(dt))
                display = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
                display.plot(cmap='binary')
                plt.xlabel('predicted classes', fontweight='bold', labelpad=15)
                plt.ylabel('true classes', fontweight='bold', labelpad=15)
                plt.savefig("./results/001_classifier_" + str(i) + "_confusion_matrix" + ".svg", bbox_inches="tight")
                plt.savefig("./results/001_classifier_" + str(i) + "_confusion_matrix" + ".pdf", bbox_inches="tight")
                plt.close()

            for i, comb in enumerate(eval_combiner.get_instances()):
                cm = sk.metrics.confusion_matrix(multiclass_assignments_to_labels(y_test), multiclass_assignments_to_labels(multi_comb_decision_outputs[i]))  # labels=np.arange(4)
                display = sk.metrics.ConfusionMatrixDisplay(confusion_matrix=cm)  # display_labels=np.arange(4)
                display.plot(cmap='binary')
                # plt.title(comb.SHORT_NAME)
                plt.xlabel('predicted classes', fontweight='bold', labelpad=15)
                plt.ylabel('true classes', fontweight='bold', labelpad=15)
                plt.savefig("./results/002_" + str(i) + "_" + comb.SHORT_NAME + "_combiner_confusion_matrix" + ".svg", bbox_inches="tight")
                plt.savefig("./results/002_" + str(i) + "_" + comb.SHORT_NAME + "_combiner_confusion_matrix" + ".pdf", bbox_inches="tight")
                plt.close()

            performance_dict[cv_run] = curr_performance_dict

    with open('final_try_smote_144classes_6sensors_11feats_cv_multiclass_performance_scores.pickle', 'wb') as handle:
        pickle.dump(overall_k_1_fold_cv_results, handle, protocol=pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    pickle_file_names = ['try_smote_144classes_6sensors_11feats_iter_1_results_2_1_fold_cv_lm1.pickle',
                         'try_smote_144classes_6sensors_11feats_iter_2_results_2_1_fold_cv_lm1.pickle',
                         'try_smote_144classes_6sensors_11feats_iter_3_results_2_1_fold_cv_lm1.pickle',
                         'try_smote_144classes_6sensors_11feats_iter_4_results_2_1_fold_cv_lm1.pickle',
                         'try_smote_144classes_6sensors_11feats_iter_5_results_2_1_fold_cv_lm1.pickle']
    paths = [pickle_file_name for pickle_file_name in pickle_file_names]
    main(paths=paths)
