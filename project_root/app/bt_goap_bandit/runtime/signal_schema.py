# src/bt_goap_bandit/runtime/signal_schema.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# --- Pydantic v2/v1 compatibility helpers -----------------------------------
try:
    # v2
    from pydantic import ConfigDict

    _PYD_VER = 2

    def _dump(model: BaseModel) -> Dict[str, Any]:
        return model.model_dump()

    def _validate(cls, data: Dict[str, Any]):
        # ensure forward refs resolved
        try:
            cls.model_rebuild()
        except Exception:
            pass
        return cls.model_validate(data)

except Exception:  # pragma: no cover - v1 fallback
    # v1
    _PYD_VER = 1

    class ConfigDict(dict):  # type: ignore
        pass

    def _dump(model: BaseModel) -> Dict[str, Any]:
        return model.dict()

    def _validate(cls, data: Dict[str, Any]):
        return cls.parse_obj(data)


# ------------------- Primary / Plan -------------------

class PrimaryModel(BaseModel):
    intent_family: str = Field(..., description="High-level family like 'answer' or 'clarify'")
    confidence: float = Field(..., ge=0.0, le=1.0)
    query: str = Field(...)

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class BeliefSlot(BaseModel):
    value: Optional[str] = None
    p: float = Field(0.0, ge=0.0, le=1.0)

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class PlanModel(BaseModel):
    plan_id: str
    steps: List[str] = []
    ready_to_retrieve: bool = False
    blocking_slots: List[str] = []
    readiness_score: float = Field(0.5, ge=0.0, le=1.0)
    belief: Dict[str, BeliefSlot] = {}
    clarify_reasons: List[str] = []

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


# ------------------- Bandit -------------------

class BanditFeatures(BaseModel):
    retrieval_used: Optional[str] = None
    retrieval_offtopic: Optional[bool] = None
    retrieval_contexts: Optional[int] = None
    retrieval_top_score: Optional[float] = None
    retrieval_hit_count: Optional[int] = None
    retrieval_confidence: Optional[float] = None
    retrieval_elapsed_ms: Optional[float] = None
    retrieval_accepted_step: Optional[str] = None
    clarify_forced: Optional[bool] = None

    # Set by cli.run(); pattern only enforced on v2
    if _PYD_VER == 2:
        provenance: Optional[str] = Field(
            default=None,
            description="Resolver provenance",
            pattern="^(regex|slm)$",
        )
        model_config = ConfigDict(extra="allow")
    else:
        provenance: Optional[str] = None

        class Config:
            extra = "allow"


class BanditModel(BaseModel):
    phase: Optional[str] = None
    arm: Optional[str] = None
    # v2 needs default_factory for nested models
    if _PYD_VER == 2:
        from pydantic import field_validator
        from pydantic import field_serializer
        features: BanditFeatures = Field(default_factory=BanditFeatures)
        model_config = ConfigDict(extra="ignore")
    else:
        features: BanditFeatures = Field(default_factory=BanditFeatures)

        class Config:
            extra = "ignore"


# ------------------- Retrieval -------------------

class RetrievalContext(BaseModel):
    content: Optional[str] = None
    file: Optional[str] = None
    bm_score: Optional[float] = None
    rank: Optional[int] = None
    uri: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class SlotLexGroup(BaseModel):
    slot: str
    canonical: Optional[str] = None
    matched: List[str] = []

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


class RetrievalModel(BaseModel):
    used: Optional[str] = "none"
    contexts: List[RetrievalContext] = []
    offtopic: Optional[bool] = None
    reason: Optional[str] = None
    top_score: Optional[float] = None
    hit_count: Optional[int] = None
    elapsed_ms: Optional[float] = None
    accepted_step: Optional[str] = None

    # Optional Slot-Lex telemetry
    slot_lex_expanded: Optional[bool] = None
    slot_lex_groups: Optional[List[SlotLexGroup]] = None
    expanded_query: Optional[str] = None

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


# ------------------- Evidence -------------------

class Evidence(BaseModel):
    regex_evidence: Optional[List[Dict[str, Any]]] = None
    slot_lex_header: Optional[Dict[str, Any]] = None

    # Set by cli.run(); pattern only enforced on v2
    if _PYD_VER == 2:
        provenance: Optional[str] = Field(
            default=None,
            description="Resolver provenance",
            pattern="^(regex|slm)$",
        )
        model_config = ConfigDict(extra="allow")
    else:
        provenance: Optional[str] = None

        class Config:
            extra = "allow"


# ------------------- Top-level payload -------------------

class SignalPayload(BaseModel):
    agent_role: str = Field("signal")
    primary: PrimaryModel
    plan: PlanModel
    bandit: BanditModel
    retrieval: RetrievalModel
    evidence: Optional[Evidence] = None
    policy_version: Optional[str] = None
    safety_flags: Optional[List[Any]] = None
    slm: Optional[Dict[str, Any]] = None

    if _PYD_VER == 2:
        model_config = ConfigDict(extra="ignore")
    else:
        class Config:
            extra = "ignore"


# Resolve forward refs on v2 to avoid validation errors
if _PYD_VER == 2:
    SignalPayload.model_rebuild()
