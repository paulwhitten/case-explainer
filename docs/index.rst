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
* **Correspondence metric**: Quantifies agreement between prediction and neighbors
* **Multiple indexing strategies**: K-D Tree, Ball Tree, or brute force
* **Automatic scaling**: Optional feature standardization
* **Metadata tracking**: Attach provenance data to training samples
* **Sklearn-compatible API**: Familiar interface for ML practitioners
* **Batch explanations**: Explain multiple predictions efficiently

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

   # From source (current development version)
   cd case-explainer
   pip install -e .

   # Dependencies
   pip install numpy scipy scikit-learn matplotlib pandas

Performance
-----------

Validated across multiple domains with excellent results:

* **Hardware Trojan Detection**: 99.9% correspondence, 25.7 ms/sample
* **Credit Card Fraud Detection**: 100% correspondence, 36.4 ms/sample
* **Medical Diagnosis (Breast Cancer)**: 93.3% correspondence, 25.9 ms/sample
* **Scalability**: Tested up to 200k training samples

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
   :caption: Additional Information

   citation
   license

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
