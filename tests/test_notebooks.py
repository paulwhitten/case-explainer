"""Structural validation for version-controlled tutorial notebooks."""

import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "02_breast_cancer_tutorial.ipynb"


def test_breast_cancer_notebook_uses_current_retrieval_api():
    with NOTEBOOK.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)

    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 39
    assert all(cell.get("id") for cell in notebook["cells"])
    source = "".join(
        line for cell in notebook["cells"] for line in cell.get("source", [])
    )
    assert "similarity=HiddenActivations(" in source
    assert "similarity=Blend(" in source
    assert 'output_weighting="predicted_class"' in source
    assert "HiddenActivationRetrieval" not in source
    assert "retrieval=" not in source
    assert "activation_layer=" not in source
