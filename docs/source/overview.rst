Overview
========

Introduction
------------

In general, there exist two main approaches to combine multiple classifiers in order to obtain reasonable classification
outputs for unseen samples: the classifier selection and the classifier fusion.
The latter one is followed by the presented framework and illustrated in the following figure.

.. image:: images/framework_context.svg

However, there are significant properties arising from such classifier ensembles as parameters which need to be
considered in the whole decision fusion process.
These parameters are forming a configuration which consists of:

- Decision fusion method (`combiner`)
- Classification problem
- Class assignment type
- Classification coverage


The `combiner` states an explicit method provided by the framework which should be applied on the input data set.
Available core methods are listed below.

The `classification problem` refers either to a multiclass or to a multilabel classification problem.
In the multiclass case, a sample is always classified into one class, while in the multilabel case, more than one class
may be assigned to a sample.

Pusion operates on classification data which is given by class assignments.
The class assignment describes memberships to each individual class for a sample.
A `class assignment type` is either crisp or continuous. Crisp assignments are equivalent to labels
and continuous assignments represent probabilities for each class being true.

The `classification coverage` states for each input classifier, which classes it is able to decide.
A classifier ensemble may yield a redundant, complementary or complementary-redundant coverage.

Core methods
------------

The following core decision fusion methods are supported by `pusion` and classified according the evidence resolution
they accept. With lower evidence resolution, a weaker a-priori information about individual classifiers can be taken
into account during the fusion process.
Utility-based methods do not take any further information about classifiers into account.
In cases where no evidence is available for a certain classification data set, a utility-based method is a reasonable
choice.
Evidence-based methods are recommended in cases where evidence (e.g. confusion matrices) but no training data is
available for each classifier.
Trainable methods provide the highest evidence resolution, since decision outputs are required from the ensemble for
each sample during the training phase.
Therefore, each `trainable combiner` is able to calculate any kind of evidence based on the training data and is even
able to analyse the behaviour of each classification method from the ensemble.

Utility-based methods (low evidence resolution):

- Borda Count (BC)
- Cosine Similarity (COS)
- Macro Majority Vote (MAMV)
- Micro Majority Vote (MIMV)
- Simple Average (AVG)

Evidence-based methods (medium evidence resolution):

- Naive Bayes (NB)
- Weighted Voting (WV)

Trainable methods (highest evidence resolution):

- Behaviour Knowledge Space (BKS)
- Decision Templates (DT)
- Decision Trees (DTree)
- Dempster Shafer (DS)
- Maximum Likelihood (MLE)
- Neural Network (NN)




Input and output
----------------

Auto fusion
----------------

Generic fusion
--------------

Other functionalities
---------------------

Evaluation
----------
