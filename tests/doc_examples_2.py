import pusion as p
from pusion.util.generator import *
import sklearn

def main():
    # Create an ensemble of 3 neural networks with different hyperparameters
    classifiers = [
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100,)),
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50)),
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50, 25)),
    ]

    # Create a random complementary-redundant classification coverage with 60% overlap.
    coverage = generate_classification_coverage(n_classifiers=3, n_classes=5, overlap=.6, normal_class=True)

    # Generate samples for the complementary-redundant ensemble
    y_ensemble_valid, y_valid, y_ensemble_test, y_test = generate_multilabel_cr_ensemble_classification_outputs(
        classifiers=classifiers,
        n_classes=5,
        n_samples=2000,
        coverage=coverage)

    # Initialize the general framework interface
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))

    # Since we are dealing with a CR output, we need to propagate the coverage to the `DecisionProcessor`.
    dp.set_coverage(coverage)

    # Train the AutoCombiner with the validation dataset
    # set the evaluation metric of the AutoFusion approach to the F1 score weighted by class-frequency
    dp.train(y_ensemble_valid, y_valid, eval_metric=p.PerformanceMetric.WEIGHTED_F1_SCORE)

    # Fuse the ensemble classification outputs (test dataset)
    y_comb = dp.combine(y_ensemble_test)

    # ----------------------------------------------------------------------------------------------------------------------

    # Define classification performance metrics used for the evaluation
    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.WEIGHTED_F1_SCORE,
        p.PerformanceMetric.MICRO_PRECISION
    ]

    # Evaluate ensemble classifiers
    eval_classifiers = p.Evaluation(*eval_metrics)
    eval_classifiers.set_instances("Ensemble")
    eval_classifiers.evaluate_cr_decision_outputs(y_test, y_ensemble_test, coverage)
    print(eval_classifiers.get_report())

    # Evaluate the fusion
    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate_cr_decision_outputs(y_test, y_comb)

    dp.set_evaluation(eval_combiner)
    print(dp.report())


if __name__ == '__main__':
    main()
