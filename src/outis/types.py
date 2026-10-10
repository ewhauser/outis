from dataclasses import dataclass

ENTITY_TYPES = (
    "NAME",
    "DATE",
    "ADDRESS",
    "PHONE",
    "EMAIL",
    "MRN",
    "SSN",
    "ACCOUNT",
    "ID",
    "URL",
    "IP",
    "AGE",
    "LOCATION",
    "FACILITY",
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    score: float = 1.0
    source: str = "model"

    def __post_init__(self):
        if self.start < 0 or self.end <= self.start or self.label not in ENTITY_TYPES:
            raise ValueError("Invalid PHI span")

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "score": self.score,
            "source": self.source,
        }
