# Case-Explainer Documentation

This directory contains the Sphinx documentation for case-explainer.

## Building the Documentation

```bash
# Build HTML documentation
make html

# Clean build artifacts
make clean

# View documentation (after building)
python -m http.server 8000 --directory _build/html
# Then open http://localhost:8000 in your browser
```

## Documentation Structure

- `index.rst` - Main documentation homepage
- `conf.py` - Sphinx configuration
- `api/` - API reference documentation
  - `explainer.rst` - CaseExplainer class documentation
  - `explanation.rst` - Explanation and Neighbor classes
  - `metrics.rst` - Metrics and correspondence functions
- `citation.rst` - Citation information
- `license.rst` - License information

## What's Generated

The documentation automatically extracts information from:

1. **Docstrings** - All docstrings in the code are parsed and formatted
2. **Type hints** - Function signatures with proper type annotations
3. **Examples** - Code examples embedded in the RST files
4. **Cross-references** - Links to related classes and functions

## Features

- **Searchable** - Full-text search of all documentation
- **Professional theme** - ReadTheDocs theme for clean, modern look
- **Cross-linking** - Links to NumPy, scikit-learn, pandas documentation
- **Responsive** - Works on desktop, tablet, and mobile
- **Code highlighting** - Syntax highlighting for all code examples

## Requirements

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
```

## Hosting Options

For production deployment:

1. **ReadTheDocs** - Free hosting for open source projects
2. **GitHub Pages** - Host from your repository
3. **Self-hosted** - Deploy `_build/html` to any web server
