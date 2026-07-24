"""Shared types for the intelligence layer."""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional
from labi.intelligence.trace import DecisionTrace

class TaskCategory(Enum):
    CODING = "coding"
    RESEARCH = "research"
    AUTOMATION = "automation"
    UTILITY = "utility"
    CONVERSATION = "conversation"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class RecommendedStrategy(Enum):
    REUSE = "reuse"
    ADAPT = "adapt"
    GENERATE_VERIFY = "generate_verify"

@dataclass
class TaskProfile:
    category: TaskCategory
    complexity: float
    risk: RiskLevel
    required_role: str
    recommended_strategy: RecommendedStrategy
    estimated_tokens: int
    keywords: List[str] = field(default_factory=list)

@dataclass
class ReuseDecision:
    action: RecommendedStrategy
    confidence: float
    reasoning: str
    similar_memory_id: Optional[str] = None
    provider_hint: Optional[str] = None
    trace: Optional[DecisionTrace] = None
