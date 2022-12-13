import time
import warnings

import pandas as pd
from sklearn.neural_network import MLPClassifier

import pusion as p
from pusion.evaluation.evaluation import Evaluation

def main():

    # display options
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # halt on error
    warnings.filterwarnings('error')


    # ---- Classifiers -----------------------------------------------------------------------------------------------------
    classifiers = [
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10,)),
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10)),
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(10, 10, 10)),
    ]

    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_ensemble_classification_outputs(
        classifiers, n_classes=5, n_samples=10000)


    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.MICRO_F1_SCORE,
        p.PerformanceMetric.MICRO_PRECISION
    ]


    print("============= Ensemble ===============")
    eval_classifiers = Evaluation(*eval_metrics)
    eval_classifiers.set_instances(classifiers)
    eval_classifiers.evaluate(y_test, y_ensemble_test)
    print(eval_classifiers.get_report())

    # ---- GenericCombiner -------------------------------------------------------------------------------------------------
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.set_parallel(False)

    t_begin = time.perf_counter()
    dp.train(y_ensemble_valid, y_valid)
    t_elapsed_train = time.perf_counter() - t_begin

    t_begin = time.perf_counter()
    dp.combine(y_ensemble_test)
    t_elapsed_combine = time.perf_counter() - t_begin

    # ---- Evaluate all combiners
    eval_combiner = Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.set_runtimes(dp.get_multi_combiner_runtimes())
    eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())

    dp.set_evaluation(eval_combiner)
    print(dp.report())

    # ---- Report runtimes
    print("------------- Runtimes ---------------")
    print(eval_combiner.get_runtime_report())
    print("Total train time:", t_elapsed_train)
    print("Total combine time:", t_elapsed_combine)


    # ---- AutoCombiner ----------------------------------------------------------------------------------------------------
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
    dp.set_parallel(False)

    t_begin = time.perf_counter()
    dp.train(y_ensemble_valid, y_valid)
    t_elapsed_train = time.perf_counter() - t_begin

    t_begin = time.perf_counter()
    y_comb = dp.combine(y_ensemble_test)
    t_elapsed_combine = time.perf_counter() - t_begin

    # ---- Evaluate AutoCombiner
    eval_combiner = Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)

    dp.set_evaluation(eval_combiner)
    print(dp.report())

    # ---- Report runtimes
    print("------------- Runtimes ---------------")
    print("Total train time:", t_elapsed_train)
    print("Total combine time:", t_elapsed_combine)


if __name__ == '__main__':
    main()

