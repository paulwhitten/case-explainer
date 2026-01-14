# Case-Explainer Assets

This directory contains datasets used in case-explainer examples and benchmarks.

## Datasets

### Hardware Trojan Detection (`hardware_trojan.csv`)

**Source**: Trust-Hub benchmark circuits with trojan insertions  
**Size**: ~735 KB (56,959 circuits)  
**Features**: 5 circuit-level metrics (LGFi, ffi, ffo, PI, PO)  
**Target**: `Trojan` (binary: 0=trojan-free, 1=trojan)  
**Format**: CSV with header row

**Usage**:
```python
import pandas as pd

# Load dataset
df = pd.read_csv('assets/data/hardware_trojan.csv')
X = df.drop('Trojan', axis=1).values
y = df['Trojan'].values
```

**Git LFS**: This file is tracked using Git Large File Storage (LFS) to keep the repository lightweight while providing easy access to the dataset.

## Other Datasets

The following datasets are used in examples but loaded from sklearn or require separate download:

- **Iris**: `sklearn.datasets.load_iris()`
- **Wine**: `sklearn.datasets.load_wine()`
- **Breast Cancer**: `sklearn.datasets.load_breast_cancer()`
- **Digits**: `sklearn.datasets.load_digits()`
- **MNIST**: `sklearn.datasets.fetch_openml('mnist_784')`
- **Credit Card Fraud**: Requires `creditcard.csv` from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Adding New Datasets

To add a new dataset to Git LFS:

```bash
# Make sure Git LFS is initialized
git lfs install

# Add your dataset
cp your_dataset.csv assets/data/
git add assets/data/your_dataset.csv
git commit -m "Add your_dataset to assets"
```

Large CSV files (>100 KB) in `assets/data/` are automatically tracked by Git LFS via `.gitattributes`.
