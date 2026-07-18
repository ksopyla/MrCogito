"""Graph-first synthetic deductive-stories dataset pipeline.

See docs/engineering_specs/deductive_stories_synthetic_dataset.md.
"""

from data.deductive_stories.schema import (
    DeductiveExample,
    EventGraph,
    GENERATION_VERSION,
    normalize_answer,
)

__all__ = [
    "DeductiveExample",
    "EventGraph",
    "GENERATION_VERSION",
    "normalize_answer",
]
