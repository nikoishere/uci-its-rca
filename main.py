"""
ActivitySim post-run RCA agent — CLI entrypoint.

Commands
--------
  analyze   Parse a simulation log and generate a root cause report.
  ingest    Add a runbook or incident report to the RAG knowledge base.
  init-db   Create the pgvector schema (run once after docker-compose up).

Examples
--------
  python main.py analyze run.log --config-dir ./configs --output report.md
  python main.py analyze run.log --no-rag
  python main.py ingest runbook.md --type runbook --title "OOM on zone skims"
  python main.py init-db
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
def cli() -> None:
    """ActivitySim post-run Root Cause Analysis agent."""


# ── analyze ───────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("log_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config-dir", "-c",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing ActivitySim YAML config files.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the report as a Markdown file at this path.",
)
@click.option(
    "--no-rag",
    is_flag=True,
    default=False,
    help="Skip the RAG knowledge-base lookup (useful when DB is not running).",
)
@click.option(
    "--top-k",
    default=4,
    show_default=True,
    help="Maximum number of RAG snippets to retrieve.",
)
def analyze(
    log_file: Path,
    config_dir: Path | None,
    output: Path | None,
    no_rag: bool,
    top_k: int,
) -> None:
    """Parse a simulation log and generate a root cause report."""
    from config import settings

    if not settings.OPENAI_API_KEY:
        console.print(
            "[red]Error:[/red] OPENAI_API_KEY is not set. "
            "Copy .env.example → .env and add your key."
        )
        sys.exit(1)

    from rca.log_parser import LogParser
    from rca.yaml_extractor import YAMLExtractor
    from rca.agent import RCAAgent
    from rca.metrics import MetricsTracker
    from rca.rate_limiter import TokenBucket
    from rca.report import print_report, save_report_markdown

    metrics = MetricsTracker()
    rate_limiter = TokenBucket(rate_per_minute=settings.RPM_LIMIT)

    # ── 1. Parse the log ──────────────────────────────────────────────────────
    console.print(f"[dim]Parsing log:[/dim] {log_file}")
    parse_result = LogParser().parse(log_file)

    if not parse_result.found:
        console.print("[yellow]No failure signal detected in the log.[/yellow]")
    else:
        console.print(
            f"[green]Failure detected:[/green] "
            f"{parse_result.failure_type} — "
            f"{parse_result.failure_line[:120]}"
        )

    # ── 2. Extract YAML config context ────────────────────────────────────────
    yaml_contexts = []
    if config_dir:
        console.print(f"[dim]Extracting YAML context from:[/dim] {config_dir}")
        yaml_contexts = YAMLExtractor().extract(
            config_dir, log_context=parse_result.context_window
        )
        console.print(f"[dim]{len(yaml_contexts)} config file(s) found[/dim]")

    # ── 3. RAG retrieval ──────────────────────────────────────────────────────
    rag_citations = []
    if not no_rag and parse_result.found:
        try:
            from rag.retriever import Retriever

            console.print("[dim]Querying knowledge base…[/dim]")
            rag_citations = Retriever(top_k=top_k).retrieve(parse_result)
            if rag_citations:
                console.print(
                    f"[dim]{len(rag_citations)} relevant snippet(s) retrieved[/dim]"
                )
            else:
                console.print("[dim]No sufficiently similar snippets found[/dim]")
        except Exception as exc:
            console.print(
                f"[yellow]RAG retrieval skipped[/yellow] ({exc}). "
                "Run with [bold]--no-rag[/bold] to suppress this warning, "
                "or run [bold]python main.py init-db[/bold] to set up the DB."
            )

    # ── 4. LLM analysis ───────────────────────────────────────────────────────
    console.print("[dim]Calling LLM for analysis…[/dim]")
    agent = RCAAgent(metrics_tracker=metrics, rate_limiter=rate_limiter)
    report = agent.analyze(parse_result, yaml_contexts, rag_citations)

    # ── 5. Display ────────────────────────────────────────────────────────────
    print_report(report, metrics.session)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    if output:
        save_report_markdown(report, output)
        console.print(f"\n[green]Report saved to:[/green] {output}")


# ── ingest ────────────────────────────────────────────────────────────────────

@cli.command("ingest")
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--type", "doc_type",
    required=True,
    type=click.Choice(["runbook", "incident"]),
    help="Document type.",
)
@click.option(
    "--title",
    required=True,
    help="Short descriptive title (shown in citations).",
)
def ingest_cmd(file_path: Path, doc_type: str, title: str) -> None:
    """Ingest a runbook or incident report into the RAG knowledge base."""
    from rag.ingest import Ingestor

    console.print(
        f'[dim]Ingesting:[/dim] {file_path} as {doc_type} — "{title}"'
    )
    doc_id = Ingestor().ingest(file_path, doc_type, title)
    console.print(f"[green]Ingested.[/green] doc_id: {doc_id}")


# ── init-db ───────────────────────────────────────────────────────────────────

@cli.command("init-db")
def init_db_cmd() -> None:
    """Create the pgvector extension, table, and index (idempotent)."""
    from rag.db import init_db

    console.print("[dim]Initialising database…[/dim]")
    init_db()
    console.print("[green]Database ready.[/green]")


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
