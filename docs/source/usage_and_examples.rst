Usage and Examples
==================


A simple example
----------------

The following code shows an illustrative and simple example of using pusion for decision outputs of three classifiers.

.. code:: python

    import pusion as p
    import numpy as np

    # Create exemplary classification outputs (class assignments)
    classifier_a = [[0, 0, 1], [0, 0, 1], [0, 1, 0]]
    classifier_b = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    classifier_c = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]

    # Create a numpy tensor
    ensemble_out = np.array([classifier_a, classifier_b, classifier_c])

    # Initialize the general framework interface
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.MACRO_MAJORITY_VOTE,
                                             problem=p.Problem.MULTI_CLASS,
                                             assignment_type=p.AssignmentType.CRISP,
                                             coverage_type=p.CoverageType.REDUNDANT))

    # Fuse the ensemble classification outputs
    fused_decisions = np.array(dp.combine(ensemble_out))

    print(fused_decisions)

Output:

.. code:: bash

    [[0 0 1]
     [0 1 0]
     [0 1 0]]


A richer example
----------------

In this example, an ensemble is created using `sklearn`'s neural network classifiers.
The 200 classification outputs are split up into validation and test datasets.
``y_ensemble_valid`` and ``y_ensemble_test`` holds the classification outputs of the whole ensemble, while
``y_valid`` and ``y_test`` are representing true labels.
The validation datasets are used to train the `DempsterShaferCombiner` combiner (:ref:`DS <ds-cref>`), while the
final fusion is performed on the test dataset (without true labels).

.. code:: python

    import pusion as p

    import sklearn

    # Create an ensemble of 3 neural networks with different hyperparameters
    classifiers = [
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,)),
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 50)),
        sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, 50, 25)),
    ]

    # Generate samples for the ensemble
    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multiclass_ensemble_classification_outputs(
        classifiers=classifiers,
        n_classes=5,
        n_samples=200)

    # User defined configuration
    conf = p.Configuration(
        method=p.Method.DEMPSTER_SHAFER,
        problem=p.Problem.MULTI_CLASS,
        assignment_type=p.AssignmentType.CRISP,
        coverage_type=p.CoverageType.REDUNDANT
    )

    # Initialize the general framework interface
    dp = p.DecisionProcessor(conf)

    # Train the selected Dempster Shafer combiner with the validation dataset
    dp.train(y_ensemble_valid, y_valid)

    # Fuse the ensemble classification outputs (test dataset)
    y_comb = dp.combine(y_ensemble_test)


Evaluation
----------

In addition to the previous example, we are able to evaluate both, the ensemble and the combiner classification
performance using the evaluation methods provided by the framework.
The critical point for achieving a reasonable comparison is obviously the usage of the same test dataset
for the combiner as well as for the ensemble.

.. code:: python

    # Define classification performance metrics used for the evaluation
    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.MICRO_F1_SCORE,
        p.PerformanceMetric.MICRO_PRECISION
    ]

    print("============= Ensemble ===============")
    eval_classifiers = p.Evaluation(*eval_metrics)
    eval_classifiers.set_instances(classifiers)
    eval_classifiers.evaluate(y_test, y_ensemble_test)
    print(eval_classifiers.get_report())

    print("============== Combiner ==============")
    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)
    print(eval_combiner.get_report())

Output:

.. code:: text

    ============= Ensemble ===============
                                         accuracy     f1  precision
    MLPClassifier [0]                       0.810  0.810      0.810
    MLPClassifier [1]                       0.800  0.800      0.800
    MLPClassifier [2]                       0.792  0.792      0.792
    ============== Combiner ==============
                                         accuracy     f1  precision
    DempsterShaferCombiner                  0.816  0.816      0.816


Auto Combiner
-------------

The following code shows an exemplary usage and evaluation of the :ref:`AutoCombiner <ac-cref>` specified in
the configuration.

.. code:: python

    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))
    dp.train(y_ensemble_valid, y_valid)
    y_comb = dp.combine(y_ensemble_test)

    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiner())
    eval_combiner.evaluate(y_test, y_comb)

    dp.set_evaluation(eval_combiner)
    print(dp.report())

Output:

.. code:: text

    ================================= AutoCombiner - Report ==================================
                       Problem: MULTI_CLASS
               Assignment type: CRISP
                 Coverage type: REDUNDANT
       Combiner type selection: UtilityBasedCombiner, TrainableCombiner
          Compatible combiners: CosineSimilarityCombiner, MacroMajorityVoteCombiner, MicroMajorityVoteCombiner, SimpleAverageCombiner, BehaviourKnowledgeSpaceCombiner, DecisionTemplatesCombiner, KNNCombiner, DempsterShaferCombiner, MaximumLikelihoodCombiner, NaiveBayesCombiner, NeuralNetworkCombiner, WeightedVotingCombiner
              Optimal combiner: CosineSimilarityCombiner
    Classification performance:
                                         accuracy  micro_f1  micro_precision
    AutoCombiner                            0.836     0.836            0.836
    ==========================================================================================

Generic Combiner
----------------

For the given data sets one could also use the :ref:`GenericCombiner <gc-cref>` to gain an overview over applicable
methods and their respective performances.

.. code:: python

    dp = p.DecisionProcessor(p.Configuration(method=p.Method.GENERIC))
    dp.train(y_ensemble_valid, y_valid)
    dp.combine(y_ensemble_test)

    eval_combiner = p.Evaluation(*eval_metrics)
    eval_combiner.set_instances(dp.get_combiners())
    eval_combiner.evaluate(y_test, dp.get_multi_combiner_decision_output())

    dp.set_evaluation(eval_combiner)
    print(dp.report())

.. note::

    The `DecisionProcessor` provides ``get_multi_combiner_decision_output()`` to retrieve fused decisions from each
    applicable combiner.

Output:

.. code:: text

    ================================ GenericCombiner - Report ================================
                       Problem: MULTI_CLASS
               Assignment type: CRISP
                 Coverage type: REDUNDANT
       Combiner type selection: UtilityBasedCombiner, TrainableCombiner
          Compatible combiners: CosineSimilarityCombiner, MacroMajorityVoteCombiner, MicroMajorityVoteCombiner, SimpleAverageCombiner, BehaviourKnowledgeSpaceCombiner, DecisionTemplatesCombiner, KNNCombiner, DempsterShaferCombiner, MaximumLikelihoodCombiner, NaiveBayesCombiner, NeuralNetworkCombiner, WeightedVotingCombiner
              Optimal combiner: WeightedVotingCombiner
    Classification performance:
                                         accuracy  micro_f1  micro_precision
    CosineSimilarityCombiner                0.836     0.836            0.836
    MacroMajorityVoteCombiner               0.836     0.836            0.836
    MicroMajorityVoteCombiner               0.836     0.836            0.836
    SimpleAverageCombiner                   0.836     0.836            0.836
    BehaviourKnowledgeSpaceCombiner         0.822     0.831            0.840
    DecisionTemplatesCombiner               0.836     0.836            0.836
    KNNCombiner                             0.826     0.836            0.846
    DempsterShaferCombiner                  0.836     0.836            0.836
    MaximumLikelihoodCombiner               0.834     0.834            0.834
    NaiveBayesCombiner                      0.836     0.836            0.836
    NeuralNetworkCombiner                   0.826     0.832            0.838
    WeightedVotingCombiner                  0.836     0.836            0.836
    ==========================================================================================

CR classification
-----------------

In `complementary-redundant` classification (CR), ensemble classifiers are not able to make predictions for all
available classes. They may complement each other or share some classes. In such cases, a `coverage` needs to be
specified in order to use the framework properly. The coverage describes for each ensemble classifier, which classes
it is able to make predictions for. In pusion, it can be defined by a simple 2D list, e.g., ``[[0,1], [0,2,3]]``, where
the first classifier is covering the classes `0,1` while the second one covers `0,2,3`.
The following code example shows how to generate and combine such complementary-redundant classification outputs.

.. code:: python

    import pusion as p
    import sklearn

    # Create an ensemble of 3 neural networks with different hyperparameters
    classifiers = [
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100,)),
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50)),
        sklearn.neural_network.MLPClassifier(max_iter=5000, hidden_layer_sizes=(100, 50, 25)),
    ]

    # Create a random complementary-redundant classification coverage with 60% overlap.
    coverage = p.generate_classification_coverage(n_classifiers=3, n_classes=5, overlap=.6, normal_class=True)

    # Generate samples for the complementary-redundant ensemble
    y_ensemble_valid, y_valid, y_ensemble_test, y_test = p.generate_multilabel_cr_ensemble_classification_outputs(
        classifiers=classifiers,
        n_classes=5,
        n_samples=2000,
        coverage=coverage)

    # Initialize the general framework interface
    dp = p.DecisionProcessor(p.Configuration(method=p.Method.AUTO))

    # Since we are dealing with a CR output, we need to propagate the coverage to the `DecisionProcessor`.
    dp.set_coverage(coverage)

    # Train the AutoCombiner with the validation dataset
    dp.train(y_ensemble_valid, y_valid)

    # Fuse the ensemble classification outputs (test dataset)
    y_comb = dp.combine(y_ensemble_test)

The framework provides also a specific evaluation methodology for complementary-redundant results.

.. code:: python

    # Define classification performance metrics used for the evaluation
    eval_metrics = [
        p.PerformanceMetric.ACCURACY,
        p.PerformanceMetric.MICRO_F1_SCORE,
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


Output:

.. code:: text

                                         accuracy  micro_f1  micro_precision
    Ensemble                                0.804     0.804            0.804
    ================================= AutoCombiner - Report ==================================
                       Problem: MULTI_LABEL
               Assignment type: CRISP
                 Coverage type: COMPLEMENTARY_REDUNDANT
       Combiner type selection: UtilityBasedCombiner, TrainableCombiner
          Compatible combiners: CRCosineSimilarity, CRMicroMajorityVoteCombiner, CRSimpleAverageCombiner, CRDecisionTemplatesCombiner, CRKNNCombiner, CRNeuralNetworkCombiner
              Optimal combiner: CRDecisionTemplatesCombiner
    Classification performance:
                                         accuracy  micro_f1  micro_precision
    AutoCombiner                            0.813     0.813            0.813
    ==========================================================================================


.. warning::
    Combiner output is always redundant, which means that all classes are covered for each sample.
    To make a reasonable comparison between the combiner and the ensemble use ``evaluate_cr_*`` methods for both.