import pusion as p
from pusion.util.generator import *
import sklearn


def start_example():
    # Create an ensemble of 3 neural networks with different hyperparameters
    classifiers = [
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,)),
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 50)),
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 50, 25)),
    ]

    # Generate samples for the ensemble
    y_ensemble_valid, y_valid, y_ensemble_test, y_test = generate_multiclass_ensemble_classification_outputs(
        classifiers=classifiers,
        n_classes=5,
        n_samples=2000,
        parallelize=False)

    # User defined configuration
    conf = p.Configuration(
        method=p.Method.DEMPSTER_SHAFER,
        problem=p.Problem.MULTI_CLASS,
        assignment_type=p.AssignmentType.CRISP,
        coverage_type=p.CoverageType.REDUNDANT
    )

    # Initialize the general framework interface
    dp = p.DecisionProcessor(conf)

    # Train the selected BehaviourKnowledgeSpace combiner with the validation data set
    dp.train(y_ensemble_valid, y_valid)

    # Fuse the ensemble classification outputs (test data set)
    y_comb = dp.combine(y_ensemble_test)

    # ----------------------------------------------------------------------------------------------------------------------

    # Define classification performance metrics used for the evaluation
    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.MICRO_F1_SCORE,
        p.PerformanceMetric.MICRO_PRECISION
    ]

    print("============= Ensemble ===============")
    # Initialize the evaluation
    eval_classifiers = p.Evaluation(*eval_metrics)
    # Set all instances to be evaluated (i.e. classifiers)
    eval_classifiers.set_instances(classifiers)
    # Evaluate
    eval_classifiers.evaluate(y_test, y_ensemble_test)
    # Report
    print(eval_classifiers.get_report())

    print("============== Combiner ==============")
    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)
    print(eval_combiner.get_report())

    # ----------------------------------------------------------------------------------------------------------------------

    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)

    dp.set_evaluation(eval_combiner)
    print(dp.report())

    # ----------------------------------------------------------------------------------------------------------------------

    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.train(y_ensemble_valid, y_valid)
    dp.combine(y_ensemble_test)

    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())

    dp.set_evaluation(eval_combiner)
    print(dp.report(eval_metric=p.PerformanceMetric.MICRO_F1_SCORE))


if __name__ == '__main__':
    start_example()