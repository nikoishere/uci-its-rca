"""
Shared data models used across the rca and rag packages.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class LogParseResult(BaseModel):
    """Output of LogParser.parse()."""

    found: bool
    failure_type: str = ""
    failure_line: str = ""
    context_window: list[str] = Field(default_factory=list)
    stack_trace: str = ""
    log_path: str = ""


class YAMLContext(BaseModel):
    """A parsed YAML config file with the keys most relevant to the failure."""

    config_path: str
    relevant_keys: dict = Field(default_factory=dict)
    raw_snippet: str = ""


class RCAOutput(BaseModel):
    """Structured output schema returned by the LLM.

    Used directly as the response_format for OpenAI Structured Outputs.
    Field descriptions are forwarded to the model as schema annotations.
    """

    failure_summary: str = Field(
        description="One sentence that describes what failed and where."
    )
    root_cause: str = Field(
        description=(
            "Detailed explanation of why the failure occurred. "
            "Reference specific variables, keys, or config values when visible."
        )
    )
    affected_component: str = Field(
        description="The ActivitySim submodel, Python module, or file that raised the error."
    )
    suggested_fixes: list[str] = Field(
        description=(
            "Ordered list of concrete, actionable fix steps. "
            "Put the most likely fix first."
        )
    )
    config_issues: list[str] = Field(
        description=(
            "YAML config problems that likely contributed to the failure. "
            "Cite the specific key and file name. Empty list if none."
        )
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description=(
            "HIGH if the stack trace clearly identifies the cause. "
            "MEDIUM if inferred from partial evidence. "
            "LOW if mostly speculative."
        )
    )


class RAGCitation(BaseModel):
    """A single retrieved snippet from the knowledge base."""

    doc_id: str
    title: str
    doc_type: str  # "runbook" | "incident"
    snippet: str
    s3_key: str
    similarity: float


class RCAReport(BaseModel):
    """Final report delivered to the researcher."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    log_path: str
    failure_summary: str
    root_cause: str
    affected_component: str
    stack_trace: str
    suggested_fixes: list[str]
    config_issues: list[str]
    rag_citations: list[RAGCitation] = Field(default_factory=list)
    confidence: Literal["HIGH", "MEDIUM", "LOW"]


class CallMetrics(BaseModel):
    """Metrics for a single LLM or embedding API call."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class SessionMetrics(BaseModel):
    """Aggregated metrics for an entire analyze session."""

    calls: list[CallMetrics] = Field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_latency_s(self) -> float:
        return sum(c.latency_s for c in self.calls)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def summary(self) -> dict:
        return {
            "calls": len(self.calls),
            "total_latency_s": round(self.total_latency_s, 2),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 4),
        }
