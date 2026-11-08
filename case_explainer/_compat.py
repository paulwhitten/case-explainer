"""Compatibility contract for constructor parameters removed in 0.3.0."""

REMOVAL_VERSION = "0.3.0"

#: Removed ``CaseExplainer`` keyword arguments mapped to their replacement.
REMOVED_PARAMETERS = {
    "activation_extractor": (
        "use similarity=CustomActivations(model=..., extractor=...) "
        "(or similarity=HiddenActivations(model=...) for an sklearn MLP)"
    ),
    "activation_layer": "use similarity=HiddenActivations(model=..., layer=...)",
    "blend_alpha": "use similarity=Blend(strategy, features=...)",
    "use_output_weights": (
        'use similarity=HiddenActivations(..., output_weighting="none") '
        "to disable output weighting"
    ),
    "retrieval": (
        "use similarity= with HiddenActivations, CustomActivations, TreeLeaf, "
        "ForestProximity, or Blend"
    ),
}


def removed_parameter_error(kwargs) -> TypeError:
    """Build the ``TypeError`` for unexpected ``CaseExplainer`` keyword arguments.

    Removed parameters get a migration hint; anything else gets the standard
    unexpected-keyword message.
    """
    for name in kwargs:
        if name in REMOVED_PARAMETERS:
            return TypeError(
                f"CaseExplainer() no longer accepts {name!r} (removed in "
                f"{REMOVAL_VERSION}); {REMOVED_PARAMETERS[name]}. "
                "See docs/migration.rst."
            )
    name = next(iter(kwargs))
    return TypeError(f"CaseExplainer() got an unexpected keyword argument {name!r}")
