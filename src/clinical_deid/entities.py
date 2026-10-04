"""Extensible API entities, independent of the trained model's label vocabulary."""

import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    score: float = 1.0
    source: str = "model"
    entity_id: str | None = None

    def __post_init__(self):
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end <= self.start
            or not isinstance(self.label, str)
            or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", self.label)
            or isinstance(self.score, bool)
            or not isinstance(self.score, (int, float))
            or not math.isfinite(self.score)
            or not 0 <= self.score <= 1
            or not isinstance(self.source, str)
            or len(self.source) > 128
            or (
                self.entity_id is not None
                and (
                    not isinstance(self.entity_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.entity_id)
                )
            )
        ):
            raise ValueError("Invalid entity")

    def as_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def checked_entities(text, spans):
    """Validate detector output before exposing it or applying a policy."""
    if not isinstance(spans, (list, tuple)) or len(spans) > 20_000:
        raise ValueError("Invalid entity output")
    result = []
    for span in spans:
        try:
            entity = Entity(
                span.start,
                span.end,
                span.label,
                span.score,
                span.source,
                getattr(span, "entity_id", None),
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("Invalid entity output") from None
        if entity.end > len(text):
            raise ValueError("Invalid entity output")
        result.append(entity)
    result.sort(key=lambda item: (item.start, item.end))
    if any(a.end > b.start for a, b in zip(result, result[1:])):
        raise ValueError("Overlapping detector output")
    return result
