"""Compatibility contracts scheduled for a future major API cleanup."""

LEGACY_ACTIVATION_PARAMETERS = (
    "activation_extractor",
    "activation_layer",
    "blend_alpha",
    "use_output_weights",
)
LEGACY_ACTIVATION_REMOVAL_VERSION = "0.3.0"
LEGACY_ACTIVATION_WARNING = (
    "Legacy activation constructor parameters are deprecated and will be "
    "removed in version 0.3.0; use "
    "retrieval=HiddenActivationRetrieval(...) instead."
)