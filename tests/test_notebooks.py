"""Structural validation for version-controlled tutorial notebooks."""

import ast
import json
import re
from pathlib import Path

import pytest


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


NOTEBOOKS = sorted((Path(__file__).parents[1] / "notebooks").glob("*.ipynb"))

#: API removed in 0.3.0 that a notebook must not use.
REMOVED_API = (
    "activation_extractor=",
    "activation_layer=",
    "blend_alpha=",
    "use_output_weights=True)",
    "retrieval=",
    "HiddenActivationRetrieval",
    "TreeLeafRetrieval",
    "ForestProximityRetrieval",
    "case_explainer.indexing",
)

#: Private attribute access on an explainer, e.g. ``explainer._strategy``.
PRIVATE_EXPLAINER_ACCESS = re.compile(r"explainer\._[A-Za-z]")

#: ``boxplot(..., labels=...)`` was removed from matplotlib; use ``tick_labels``
#: or ``xticks``.
BOXPLOT_LABELS = re.compile(r"boxplot\([^\n]*[(,\s]labels=")


def _code_cells(path):
    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    return [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_all_tutorial_notebooks_are_found():
    assert len(NOTEBOOKS) == 4


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_are_valid_python(path):
    for index, source in enumerate(_code_cells(path)):
        lines = [
            line
            for line in source.splitlines()
            if not line.lstrip().startswith(("%", "!"))
        ]
        try:
            ast.parse("\n".join(lines))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path.name} code cell {index}: {exc}")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_only_current_public_api(path):
    source = "\n".join(_code_cells(path))
    for removed in REMOVED_API:
        assert removed not in source, f"{path.name} uses removed API {removed!r}"
    private = PRIVATE_EXPLAINER_ACCESS.search(source)
    assert private is None, (
        f"{path.name} reads a private explainer attribute near "
        f"{source[private.start():private.start() + 60]!r}"
    )
    assert (
        BOXPLOT_LABELS.search(source) is None
    ), f"{path.name} passes labels= to boxplot, which matplotlib removed"


IMBALANCED_NOTEBOOKS = [
    path
    for path in NOTEBOOKS
    if path.name.startswith(("03_fraud", "04_hardware_trojan"))
]


@pytest.mark.parametrize("path", IMBALANCED_NOTEBOOKS, ids=lambda p: p.name)
def test_imbalanced_tutorials_demonstrate_balanced_class_weights(path):
    """The imbalanced tutorials must show the recommended weighting, with
    executed output, next to the default they use for their main path."""
    with path.open(encoding="utf-8") as notebook_file:
        notebook = json.load(notebook_file)
    balanced_cells = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and 'class_weights="balanced"' in "".join(cell.get("source", []))
    ]
    assert balanced_cells, f"{path.name} does not demonstrate class_weights='balanced'"
    outputs = [
        "".join(output.get("text", ""))
        for cell in balanced_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    ]
    assert any(
        "Identical neighbors under both weightings: True" in text for text in outputs
    )
