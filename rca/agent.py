"""
LLM-powered RCA agent.

Wraps OpenAI Structured Outputs so the response is always a validated
RCAOutput Pydantic model — no regex JSON scraping needed.

Resilience:
  - Rate limiting via TokenBucket before every call.
  - Automatic retry with exponential back-off for transient API errors
    (RateLimitError, APITimeoutError, APIConnectionError).
  - Hard timeout on the httpx client inside the OpenAI SDK.
  - Graceful fallback report when all retries are exhausted.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import openai
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings
from rca.metrics import MetricsTracker, compute_cost
from rca.models import (
    CallMetrics,
    LogParseResult,
    RCAOutput,
    RCAReport,
    RAGCitation,
    YAMLContext,
)
from rca.rate_limiter import TokenBucket

_SYSTEM_PROMPT = """\
You are an expert in ActivitySim, a Python-based activity-based travel \
demand modelling framework used in transportation planning research.

You will receive:
1. A log excerpt from a failed ActivitySim simulation run.
2. Relevant YAML configuration context (if available).
3. Snippets from past runbooks and resolved incidents (if available).

Your task: perform root cause analysis and return a structured JSON report.

Focus on these failure classes:
- Python exceptions and tracebacks (KeyError, AttributeError, ValueError, …)
- Memory errors (MemoryError, OOM kills, swap exhaustion)
- Data integrity issues (missing columns, shape mismatches, NaN propagation)
- Configuration errors (wrong paths, incompatible settings, missing keys)
- Multiprocessing / chunking failures
- File I/O errors (missing input files, permission denied, disk full)

Guidelines:
- Be specific and concise; researchers need to act fast.
- If the stack trace clearly identifies the cause, set confidence to HIGH.
- If you are inferring from partial evidence, set confidence to MEDIUM.
- If you are mostly speculating, set confidence to LOW.
- Suggested fixes must be concrete and ordered (most likely fix first).
- Config issues should cite the specific key and file name when possible.
"""


class RCAAgent:
    def __init__(
        self,
        metrics_tracker: Optional[MetricsTracker] = None,
        rate_limiter: Optional[TokenBucket] = None,
    ) -> None:
        self._client = OpenAI(
            api_key=settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_S,
            max_retries=0,  # retries handled by tenacity below
        )
        self._metrics = metrics_tracker or MetricsTracker()
        self._rate_limiter = rate_limiter or TokenBucket(
            rate_per_minute=settings.RPM_LIMIT
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def analyze(
        self,
        parse_result: LogParseResult,
        yaml_contexts: Optional[list[YAMLContext]] = None,
        rag_citations: Optional[list[RAGCitation]] = None,
    ) -> RCAReport:
        if not parse_result.found:
            return self._no_failure_report(parse_result.log_path)

        user_message = self._build_user_message(
            parse_result,
            yaml_contexts or [],
            rag_citations or [],
        )

        self._rate_limiter.wait_and_consume()

        t0 = time.monotonic()
        try:
            rca_output, usage = self._call_llm(user_message)
        except Exception as exc:
            return self._fallback_report(parse_result, str(exc))
        latency = time.monotonic() - t0

        self._metrics.record(
            CallMetrics(
                model=settings.OPENAI_MODEL,
                latency_s=round(latency, 3),
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=compute_cost(
                    settings.OPENAI_MODEL,
                    usage.prompt_tokens,
                    usage.completion_tokens,
                ),
            )
        )

        return RCAReport(
            log_path=parse_result.log_path,
            failure_summary=rca_output.failure_summary,
            root_cause=rca_output.root_cause,
            affected_component=rca_output.affected_component,
            stack_trace=parse_result.stack_trace,
            suggested_fixes=rca_output.suggested_fixes,
            config_issues=rca_output.config_issues,
            rag_citations=rag_citations or [],
            confidence=rca_output.confidence,
        )

    # ── Private ───────────────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            )
        ),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(settings.OPENAI_MAX_RETRIES),
        reraise=True,
    )
    def _call_llm(self, user_message: str) -> tuple[RCAOutput, object]:
        response = self._client.beta.chat.completions.parse(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=RCAOutput,
            temperature=0.1,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            raise ValueError("LLM returned null structured output")
        return parsed, response.usage

    def _build_user_message(
        self,
        parse_result: LogParseResult,
        yaml_contexts: list[YAMLContext],
        rag_citations: list[RAGCitation],
    ) -> str:
        parts: list[str] = []

        parts.append("## Log Excerpt\n")
        parts.append(
            f"**Failure signal:** `{parse_result.failure_type}` — "
            f"`{parse_result.failure_line}`\n\n"
        )
        # Cap at 80 lines so we stay well within context limits.
        window = parse_result.context_window[-80:]
        parts.append("```\n" + "\n".join(window) + "\n```\n")

        if yaml_contexts:
            parts.append("\n## YAML Configuration Context\n")
            for ctx in yaml_contexts:
                name = Path(ctx.config_path).name
                parts.append(f"### {name}\n```yaml\n{ctx.raw_snippet}\n```\n")

        if rag_citations:
            parts.append("\n## Relevant Runbooks & Past Incidents\n")
            for i, cit in enumerate(rag_citations, 1):
                parts.append(
                    f"### [{i}] {cit.title} ({cit.doc_type})\n"
                    f"*Source: `{cit.s3_key}` — similarity {cit.similarity:.2f}*\n"
                    f"```\n{cit.snippet}\n```\n"
                )

        return "".join(parts)

    def _no_failure_report(self, log_path: str) -> RCAReport:
        return RCAReport(
            log_path=log_path,
            failure_summary="No failure signal detected in the log.",
            root_cause=(
                "The log did not contain any recognised failure markers "
                "(Traceback, ERROR, FATAL, MemoryError, etc.)."
            ),
            affected_component="Unknown",
            stack_trace="",
            suggested_fixes=[
                "Confirm this is the correct log file for the failed run.",
                "Check whether the process was killed silently (OOM, wall-time limit).",
                "Look for a separate stderr file or SLURM/PBS job log.",
            ],
            config_issues=[],
            confidence="LOW",
        )

    def _fallback_report(self, parse_result: LogParseResult, error: str) -> RCAReport:
        return RCAReport(
            log_path=parse_result.log_path,
            failure_summary=f"LLM analysis unavailable — {error}",
            root_cause=(
                parse_result.failure_line
                or "Unknown — review the stack trace manually."
            ),
            affected_component="Unknown",
            stack_trace=parse_result.stack_trace,
            suggested_fixes=[
                "Review the stack trace above manually.",
                "Retry once the LLM service is reachable.",
            ],
            config_issues=[],
            confidence="LOW",
        )
