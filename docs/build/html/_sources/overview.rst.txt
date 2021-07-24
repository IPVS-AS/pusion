Overview
========

Introduction
------------

In general, there exist two main approaches to combine multiple classifiers in order to obtain reasonable classification
outputs for unseen samples: the classifier selection and the classifier fusion.
The latter one is followed by the presented framework and illustrated in :numref:`fig-context`.

.. _fig-context:

.. figure:: images/framework_context.svg

   Architectural embedding of the decision fusion framework

However, there are significant properties arising from such classifier ensembles as parameters which need to be
considered in the whole decision fusion process.
These parameters are forming a configuration which consists of:

- Decision fusion method (`combiner`)
- Classification problem
- Class assignment type
- Classification coverage


The `combiner` states an explicit method provided by the framework which should be applied on the input data set.
Available core methods are listed in the following section.

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

The following core decision fusion methods are supported by `pusion` and classified according to the evidence resolution
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

- Borda Count (:ref:`BC <bc-cref>`)
- Cosine Similarity (:ref:`COS <cos-cref>`)
- Macro Majority Vote (:ref:`MAMV <mamv-cref>`)
- Micro Majority Vote (:ref:`MIMV <mimv-cref>`)
- Simple Average (:ref:`AVG <avg-cref>`)

Evidence-based methods (medium evidence resolution):

- Naive Bayes (:ref:`NB <nb-cref>`)
- Weighted Voting (:ref:`WV <wv-cref>`)

Trainable methods (highest evidence resolution):

- Behaviour Knowledge Space (:ref:`BKS <bks-cref>`)
- Decision Templates (:ref:`DT <dt-cref>`)
- k Nearest Neighbors (:ref:`KNN <knn-cref>`)
- Dempster Shafer (:ref:`DS <ds-cref>`)
- Maximum Likelihood (:ref:`MLE <mle-cref>`)
- Neural Network (:ref:`NN <nn-cref>`)


Data input and output
---------------------
The input type used for classification data is generic and applies to all provided decision fusion methods.
It is given by a 3D `numpy.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_ tensor,
which is illustrated in :numref:`fig-input_tensor_illustration`.

.. _fig-input_tensor_illustration:

.. figure:: images/input_tensor_illustration.svg

   Illustration of the input tensor for a multilabel problem with crisp assignments
   (3 samples, 4 classes, 2 classifiers).

The same applies also to the pusion's return, except that the output matrix is a 2D `numpy.ndarray`.

.. note::
   In case of complementary-redundant decisions, the coverage needs to be specified besides ordinary python lists,
   which are used as an alternative to the `numpy.ndarray`.

AutoFusion
-----------
The framework provides an additional fusion method :ref:`AutoCombiner <ac-cref>` which is able to the detect the configuration
based on the input classification data and to automatically select the fusion method with the best classification
performance for the given problem.
The `AutoCombiner` bundles all methods provided by the framework and probes each of them for the application on the
given classification data.
The `AutoCombiner` is transparent to the user as each of the core fusion methods.

Generic fusion
--------------
In contrast to the `AutoCombiner`, the :ref:`GenericCombiner <gc-cref>` retrieves fusion results obtained by all
compatible core methods by means of a `numpy.ndarray` tensor. In this case, the evaluation as well as the method
selection is handed over to the user.


Further functionalities
-----------------------
- Classification data and coverage generation (see module :ref:`generator <generator-cref>`)
- Useful transformations for decision outputs, e.g. multilabel to multiclass conversion
  (see module :ref:`transformer <transformer-cref>`)
- Evaluation methods for different classification and coverage types (see class :ref:`Evaluation <eval-cref>`)