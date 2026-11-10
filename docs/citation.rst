Citation
========

If you use Case-Explainer in your research, please cite:

BibTeX
------

.. code-block:: bibtex

   @software{case_explainer2025,
     author = {Whitten, Paul and Wolff, Francis and Papachristou, Chris},
     title = {Case-Explainer: General-Purpose Case-Based Explainability},
     year = {2025},
     url = {https://github.com/paulwhitten/case-explainer}
   }

Related Publications
--------------------

The case-based explainability approach implemented in this package was validated in:

**Hardware Trojan Detection**

* Whitten, P., Wolff, F., & Papachristou, C. (2024). "Explainable Hardware Trojan Detection 
  Using Case-Based Reasoning." *Journal of Electronic Testing: Theory and Applications (JETTA)*.

That work reported 99.9% correspondence on hardware trojan detection with 56,959 samples,
using its own pipeline and test sample. The figures measured with this library's
0.3.0 validation, including results on the trojans the model misses, are in
:doc:`calibration`.

**Theoretical Foundation**

The general approach is inspired by:

* Caruana, R., Kangarloo, H., Dionisio, J., Sinha, U., & Johnson, D. (1999). 
  "Case-based explanation of non-case-based learning methods." 
  *Proceedings of the AMIA Symposium*, 212-215.

Acknowledgments
---------------

* Built with scikit-learn, NumPy, and matplotlib
* Inspired by interpretable machine learning research
* Validated across medical, financial, and security domains
