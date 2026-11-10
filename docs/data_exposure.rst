Data Exposure and Privacy
=========================

Case-explainer has no privacy-preserving features, and none are planned. An
explanation is a set of real training records. Showing them is the method, so
the library cannot hide them and still explain the prediction.

Read this page before using the library with data that not everyone may see.

What an explanation discloses
-----------------------------

Each :class:`~case_explainer.explanation.Neighbor` in an explanation carries:

* ``features``: the complete, unscaled feature values of one training row
* ``index``: that row's position in the training set
* ``label``: the row's class label
* ``distance``: its exact distance from the query in the retrieval space
* ``metadata``: every metadata field supplied for that row

``Explanation.to_dict()`` exports all of it, together with the query sample
itself. ``Explanation.summary()`` prints each neighbor's row position, label,
distance, and metadata. ``Explanation.plot()`` draws the neighbors' feature
values.

What to treat as training data
------------------------------

**Explanations.** Anyone who can read an explanation can read the training
rows in it. A neighbor whose feature values equal the query confirms that the
queried record is in the training set:

.. code-block:: python

   explanation = explainer.explain_instance(candidate_record, model=clf)
   nearest = explanation.neighbors[0]
   if np.array_equal(nearest.features, candidate_record):
       # candidate_record is in the training data, at row nearest.index
       ...

The reported distance for such a match is zero or floating-point noise close
to it.

**Saved explainers.** A pickled :class:`~case_explainer.CaseExplainer` contains
the whole training set: raw features, labels, and metadata. Store and share it
under the same controls as the data itself.

**Query access.** A service that explains arbitrary inputs lets a caller
retrieve the training rows nearest to any point they choose. Do not offer one
to people who may not see the data.

When the library is appropriate
-------------------------------

Use case-explainer when the audience for the explanations is already entitled
to see the training records, such as the analysts, clinicians, or engineers who
work with that data. This matters most for patient records, financial
transactions, security telemetry, and personal data.

If the training data cannot be shown to the people who need the explanation,
use a method that does not expose it, such as LIME or SHAP.

What remains your responsibility
--------------------------------

1. Leave direct identifiers out of the features and out of ``metadata``.
2. Pass ``return_provenance=False`` where metadata should not travel with an
   explanation.
3. Put access control and audit logging around wherever explanations and saved
   explainers are served or stored.
4. Confirm the legal basis for disclosing training records (GDPR, HIPAA, and
   similar) before deployment.

``return_provenance=False`` removes metadata only. Feature values, labels, row
positions, and distances are still returned:

.. code-block:: python

   explanation = explainer.explain_instance(
       sample, model=clf, return_provenance=False
   )
   explanation.neighbors[0].metadata   # {}
   explanation.neighbors[0].features   # still the full training row

Why there is no masking feature
-------------------------------

Hiding selected columns in the output would not make explanations private.
Row positions, labels, and near-zero distances still reveal whether a record
is in the training set, and repeated queries can narrow down the values of a
hidden row. Techniques with formal guarantees, such as differential privacy, conflict
with the purpose of the library, because the explanation is a real record.

Options that limit what is displayed may be added as a convenience. They would
be disclosure control for a trusted audience and not a privacy guarantee.
