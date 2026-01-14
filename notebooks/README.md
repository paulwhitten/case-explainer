# Case-Explainer Tutorial Notebooks

Interactive Jupyter notebooks demonstrating case-based explainability across multiple domains.

## Available Tutorials

### 1. Iris Dataset (`01_iris_tutorial.ipynb`)
**Status:** ✅ Complete

**What you'll learn:**
- Introduction to case-based explainability
- Creating and using CaseExplainer
- Understanding correspondence scores
- Analyzing prediction confidence
- Batch explanations and visualization

**Dataset:** 150 samples, 4 features, 3 classes (setosa, versicolor, virginica)

**Best for:** Getting started, understanding core concepts

### 2. Breast Cancer (`02_breast_cancer_tutorial.ipynb`)
**Status:** ✅ Complete

**Medical diagnosis domain** with 569 samples, 30 features, binary classification

### 3. Fraud Detection (`03_fraud_detection_tutorial.ipynb`)
**Status:** ✅ Complete

**Financial domain** with 284k samples, 30 features, highly imbalanced data

### 4. Hardware Trojan (`04_hardware_trojan_tutorial.ipynb`)
**Status:** ✅ Complete

**Security domain** with 56k samples, 10 features, large-scale real-world data

## Running the Notebooks

### Prerequisites

```bash
# Install case-explainer
cd ..
pip install -e .

# Install Jupyter
pip install jupyter notebook
```

### Launch Jupyter

```bash
# From the notebooks directory
jupyter notebook

# Or from project root
jupyter notebook notebooks/
```

Your browser will open with the notebook interface. Click on any `.ipynb` file to start!

## What to Expect

Each tutorial follows a similar structure:
1. **Setup** - Import libraries and load data
2. **Train Classifier** - Build a model for predictions
3. **Create Explainer** - Initialize CaseExplainer with training data
4. **Single Explanation** - Explain one prediction in detail
5. **Batch Analysis** - Explain multiple predictions
6. **Visualization** - Plot results and distributions
7. **Insights** - Analyze correspondence patterns

## Key Concepts

**Case-Based Explainability:** Instead of feature importances, you get similar training examples as evidence

**Correspondence:** Quantifies agreement between prediction and neighbors (0-100%)
- High (≥85%): Strong confidence
- Medium (70-85%): Moderate confidence  
- Low (<70%): Uncertain prediction

**Nearest Neighbors:** The k most similar training samples to the test instance

## Tips

- Run cells in order (top to bottom)
- Experiment with different k values
- Try different indexing algorithms
- Compare correspondence across datasets
- Look at low-correspondence predictions - they're interesting!

## Troubleshooting

**Import error for case_explainer:**
```python
import sys
sys.path.insert(0, '..')  # Add parent directory to path
from case_explainer import CaseExplainer
```

**Missing dataset:**
- Iris, Wine, Breast Cancer, Digits: Built into sklearn
- Fraud Detection: Requires `creditcard.csv` in project root
- Hardware Trojan: Requires data in `../explainable_hw_trojan_detection_pipeline/data/processed/`

## Contributing

Have ideas for additional tutorials? Open an issue or submit a PR!

Suggested domains:
- Text classification
- Time series
- Recommender systems
- Multi-label classification
