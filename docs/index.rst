Case-Explainer Documentation
============================

**Case-Explainer** provides model-agnostic explanations through training set precedent and nearest neighbor correspondence.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Overview
--------

While some explainability methods provide feature importance scores, case-based explainability answers: 
**"Why was this prediction made?"** by showing similar training examples.

Instead of: *"Feature X has importance 0.45"*  

You get: *"This sample is classified as X because it resembles these 5 training examples"*

Key Features
------------

* **Model-agnostic**: Works with any classifier (sklearn, XGBoost, neural networks, etc.)
* **Correspondence metric**: Quantifies agreement between prediction and neighbors,
  with configurable thresholds and ``class_weights="balanced"`` for imbalanced data
* **Pluggable k-NN backend**: sklearn's ``auto``, ``kd_tree``, ``ball_tree``, or ``brute`` via ``algorithm=``
* **Automatic scaling**: Optional feature standardization
* **Metadata tracking**: Attach provenance data to training samples
* **Sklearn-compatible API**: Familiar interface for ML practitioners
* **Batch explanations**: Vectorized ``explain_batch`` with one model call and one k-NN query per batch
* **Model-informed similarity**: Retrieve cases from a neural network's hidden activations, a decision tree's matching leaf, or random-forest proximity via ``similarity=``

Quick Start
-----------

.. code-block:: python

   from case_explainer import CaseExplainer
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier

   # Load data
   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

   # Train classifier
   clf = RandomForestClassifier()
   clf.fit(X_train, y_train)

   # Create explainer
   explainer = CaseExplainer(
       X_train=X_train,
       y_train=y_train,
       feature_names=['sepal_len', 'sepal_width', 'petal_len', 'petal_width'],
       algorithm='kd_tree'
   )

   # Explain a prediction
   explanation = explainer.explain_instance(X_test[0], k=5, model=clf)
   print(f"Correspondence: {explanation.correspondence:.2%}")
   print(explanation.summary())

Installation
------------

.. code-block:: bash

   pip install case-explainer

   # Or from source (current development version)
   git clone https://github.com/paulwhitten/case-explainer.git
   cd case-explainer
   pip install -e .

Performance
-----------

Validated across multiple domains in feature space with the 0.3.0 defaults
(single batch runs on the development machine; the imbalanced domains are
measured on test subsets that include every positive):

* **Hardware Trojan Detection**: 99.3% mean correspondence; 99.7% on correct
  predictions, 83.2% on incorrect ones; 0.06 ms/sample
* **Credit Card Fraud Detection**: 99.3% mean correspondence; 99.6% on correct
  predictions, 84.3% on incorrect ones; 0.15 ms/sample
* **Medical Diagnosis (Breast Cancer)**: 92.9% mean correspondence; 96.2% on
  correct predictions, 45.6% on incorrect ones; 0.11 ms/sample
* **Scalability**: Tested up to 200k training samples

On imbalanced data pass ``class_weights="balanced"``. :doc:`calibration` shows
how the interpretation labels behave with and without it.

**Note on Correspondence**: This metric measures agreement between predictions and retrieved 
neighbors, not prediction accuracy or quality. High correspondence indicates consistency with 
training data patterns.

Data Exposure
-------------

Explanations contain real training records: feature values, labels, row
positions, and metadata. A saved explainer contains the whole training set.
The library has no privacy-preserving mode, so use it where the audience for
the explanations may see the training data. See :doc:`data_exposure`.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/explainer
   api/explanation
   api/metrics

.. toctree::
   :maxdepth: 1
   :caption: Guides

   calibration
   data_exposure

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   citation
   migration
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
