"""Decision observability.""" 

from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DecisionTrace:
    stage: str
    signals: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)

    def add_signal(self, name: str, value: float) -> None:
        self.signals[name] = round(value, 3)

    def add_note(self, message: str) -> None:
        self.notes.append(message)

    def add_meta(self, key: str, value: str) -> None:
        self.metadata[key] = value
