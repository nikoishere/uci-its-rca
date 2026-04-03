"""
Renders the RCAReport as a Rich terminal display and/or a Markdown file.
"""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rca.models import RCAReport, SessionMetrics

console = Console()

_CONFIDENCE_COLOUR = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}


def print_report(
    report: RCAReport,
    metrics: SessionMetrics | None = None,
) -> None:
    colour = _CONFIDENCE_COLOUR.get(report.confidence, "white")
    console.print()

    # ── Header ────────────────────────────────────────────────────────────────
    console.print(
        Panel(
            f"[bold]{report.failure_summary}[/bold]\n\n"
            f"[dim]Log:[/dim]        {report.log_path}\n"
            f"[dim]Generated:[/dim]  {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"[dim]Confidence:[/dim] [{colour}]{report.confidence}[/{colour}]",
            title="[bold red]ActivitySim RCA Report[/bold red]",
            border_style="red",
        )
    )

    # ── Root cause ────────────────────────────────────────────────────────────
    console.print(Panel(report.root_cause, title="Root Cause", border_style="yellow"))

    # ── Affected component ────────────────────────────────────────────────────
    console.print(
        f"[bold]Affected component:[/bold] [cyan]{report.affected_component}[/cyan]\n"
    )

    # ── Stack trace ───────────────────────────────────────────────────────────
    if report.stack_trace:
        console.print(
            Panel(
                f"[dim]{report.stack_trace}[/dim]",
                title="Stack Trace",
                border_style="dim",
            )
        )

    # ── Suggested fixes ───────────────────────────────────────────────────────
    if report.suggested_fixes:
        console.print("\n[bold yellow]Suggested Fixes[/bold yellow]")
        for i, fix in enumerate(report.suggested_fixes, 1):
            console.print(f"  [bold]{i}.[/bold] {fix}")

    # ── Config issues ─────────────────────────────────────────────────────────
    if report.config_issues:
        console.print("\n[bold magenta]Config Issues[/bold magenta]")
        for issue in report.config_issues:
            console.print(f"  • {issue}")

    # ── RAG citations ─────────────────────────────────────────────────────────
    if report.rag_citations:
        console.print("\n[bold blue]Related Runbooks & Incidents[/bold blue]")
        for i, cit in enumerate(report.rag_citations, 1):
            console.print(
                f"  [{i}] [bold]{cit.title}[/bold] ({cit.doc_type})  "
                f"[dim]similarity {cit.similarity:.2f} — {cit.s3_key}[/dim]"
            )

    # ── Metrics ───────────────────────────────────────────────────────────────
    if metrics and metrics.calls:
        console.print()
        table = Table(
            title="Session Metrics",
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        s = metrics.summary()
        table.add_row("LLM / embedding calls", str(s["calls"]))
        table.add_row("Total latency", f"{s['total_latency_s']} s")
        table.add_row("Prompt tokens", f"{s['prompt_tokens']:,}")
        table.add_row("Completion tokens", f"{s['completion_tokens']:,}")
        table.add_row("Estimated cost", f"${s['estimated_cost_usd']:.4f}")
        console.print(table)


def save_report_markdown(report: RCAReport, output_path: Path) -> None:
    lines: list[str] = [
        "# ActivitySim RCA Report\n",
        f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Log:** `{report.log_path}`  ",
        f"**Confidence:** {report.confidence}\n",
        f"## Failure Summary\n\n{report.failure_summary}\n",
        f"## Root Cause\n\n{report.root_cause}\n",
        f"## Affected Component\n\n`{report.affected_component}`\n",
    ]

    if report.stack_trace:
        lines.append(f"## Stack Trace\n\n```\n{report.stack_trace}\n```\n")

    if report.suggested_fixes:
        lines.append("## Suggested Fixes\n")
        for i, fix in enumerate(report.suggested_fixes, 1):
            lines.append(f"{i}. {fix}")
        lines.append("")

    if report.config_issues:
        lines.append("## Config Issues\n")
        for issue in report.config_issues:
            lines.append(f"- {issue}")
        lines.append("")

    if report.rag_citations:
        lines.append("## Related Runbooks & Incidents\n")
        for i, cit in enumerate(report.rag_citations, 1):
            lines.append(
                f"{i}. **{cit.title}** ({cit.doc_type})  \n"
                f"   Source: `{cit.s3_key}` — similarity {cit.similarity:.2f}  \n"
                f"   > {cit.snippet[:400].strip()}…\n"
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")
