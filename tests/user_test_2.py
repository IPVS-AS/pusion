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


    classifiers = [
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(100,)),
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50)),
        MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50, 25)),
    ]

    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.MICRO_F1_SCORE,
        p.PerformanceMetric.MICRO_PRECISION
    ]

    coverage = p.generate_classification_coverage(n_classifiers=len(classifiers), n_classes=5, overlap=.6, normal_class=True)

    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_cr_ensemble_classification_outputs(
        classifiers=classifiers, n_classes=5, n_samples=2000, coverage=coverage)


    print("============= Ensemble ===============")
    eval_classifiers = Evaluation(*eval_metrics)
    eval_classifiers.set_instances("Ensemble")
    eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
    print(eval_classifiers.get_report())


    # ---- GenericCombiner -------------------------------------------------------------------------------------------------
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.set_coverage(coverage)
    dp.train(y_ensemble_valid, y_valid)
    dp.combine(y_ensemble_test)

    eval_combiner = Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate_cr_multi_combiner_decision_outputs(y_test, dp.get_multi_combiner_decision_output())

    dp.set_evaluation(eval_combiner)
    print(dp.report())


    # ---- AutoCombiner ----------------------------------------------------------------------------------------------------
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
    dp.set_coverage(coverage)
    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    eval_combiner = Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate_cr_decision_outputs(y_test, y_comb)

    dp.set_evaluation(eval_combiner)
    print(dp.report())


if __name__ == '__main__':
    main()