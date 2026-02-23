from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# ---------- Top-level ----------

@dataclass
class Meta:
    pipeline_version: str
    policy_version: str
    subplans_count: int
    run_id: str
    sentiment_provider: str


@dataclass
class Input:
    text: str
    intent: str
    entities: List[Any]
    sentiment_hint: Optional[str]


# ---------- Enriched / Summary ----------

@dataclass
class Summary:
    num_chunks: int
    top_chunks: List[str]
    scores:List[float]
    tags: List[str]
    context_chars: int


@dataclass
class Enriched:
    text: str
    intent: Optional[str]
    entities: List[Any]
    semantic_tags: List[str]
    retrieved_contexts: List[str]
    summary: Summary


# ---------- GOAP Plan ----------

@dataclass
class GoapReasoning:
    intent: Optional[str]
    semantic_topic: Optional[str]
    semantic_tags: List[str]
    entities: List[Any]
    required_params: List[str]
    auto_filled: Dict[str, Any]


@dataclass
class GoapPlan:
    tool: Optional[str]
    params: Dict[str, Any]
    missing: List[str]
    confidence: float
    readiness_score: float
    missing_ratio: float
    reasoning: Optional[GoapReasoning] = None

# ---------- Sentiment ----------

@dataclass
class Scores:
    label: str
    score: float


@dataclass
class RawSentiment:
    mode: str
    scores: Scores


@dataclass
class Sentiment:
    valence: str
    intensity: float
    raw: RawSentiment


# ---------- Meta Manifest ----------

@dataclass
class MetaManifest:
    intent_family: str
    behavioral_family: str
    scope: str
    tags: List[str]


# ---------- Subplan + top-level response ----------

@dataclass
class Subplan:
    subplan_id: int
    input: Input
    enriched: Optional[Enriched]
    goap_plan: GoapPlan
    sentiment: Sentiment
    bandit_arm: str
    bt_output: Optional[Any]
    meta_manifest: MetaManifest


@dataclass
class IntentAgentResponse:
    agent_role: str
    meta: Meta
    input_text: str
    subplans: List[Subplan]


# ---------- SLM NLU adapter side ----------

@dataclass
class SLMIdentifiedSubPlan:
    text: str
    intent: str
    entities: List[Any]
    sentiment_hint: Optional[str]
    confidence: float


@dataclass
class SlmNluAdapterResponse:
    text: str
    subplans: List[SLMIdentifiedSubPlan]
