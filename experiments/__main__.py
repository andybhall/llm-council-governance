"""Unified CLI for running and analyzing experiments."""

import asyncio
from pathlib import Path
from typing import Optional

import click


@click.group()
def cli():
    """LLM Council Governance Experiment Framework.

    Run experiments, analyze results, and generate reports.
    """
    pass


@cli.command()
@click.option(
    "--n-questions",
    "-n",
    default=40,
    help="Number of questions per benchmark",
)
@click.option(
    "--n-replications",
    "-r",
    default=3,
    help="Number of replications per condition",
)
@click.option(
    "--output-dir",
    "-o",
    default="experiments/results",
    help="Directory to save results",
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from existing results",
)
def run(n_questions: int, n_replications: int, output_dir: str, resume: bool):
    """Run the pilot experiment with default configuration."""
    from experiments.run_pilot import run_pilot

    click.echo(f"Running pilot experiment...")
    click.echo(f"  Questions: {n_questions} per benchmark")
    click.echo(f"  Replications: {n_replications}")
    click.echo(f"  Output: {output_dir}")
    click.echo(f"  Resume: {resume}")

    results = asyncio.run(
        run_pilot(
            n_questions=n_questions,
            n_replications=n_replications,
            output_dir=output_dir,
        )
    )

    click.echo(f"\nExperiment complete. {len(results)} results saved.")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True), default="experiments/results")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for analysis report",
)
def analyze(results_dir: str, output: Optional[str]):
    """Analyze experiment results and generate report."""
    from experiments.analyze_pilot import analyze_pilot, load_results_as_dataframe

    results_path = Path(results_dir)

    if not (results_path / "pilot_results.json").exists():
        click.echo(f"Error: No pilot_results.json found in {results_dir}")
        raise SystemExit(1)

    click.echo(f"Analyzing results from {results_dir}...")

    df = analyze_pilot(str(results_path))

    if output:
        # Copy the generated report to the specified output
        report_path = results_path / "analysis_report.txt"
        if report_path.exists():
            Path(output).write_text(report_path.read_text())
            click.echo(f"Report saved to {output}")

    click.echo(f"\nAnalysis complete. {len(df)} trials analyzed.")


@cli.command()
@click.argument("results_dir", type=click.Path(exists=True), default="experiments/results")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["png", "svg", "pdf"]),
    default="png",
    help="Output format for charts",
)
def report(results_dir: str, format: str):
    """Generate charts and visualizations from results."""
    from experiments.analyze_pilot import generate_charts

    results_path = Path(results_dir)

    if not (results_path / "pilot_results.json").exists():
        click.echo(f"Error: No pilot_results.json found in {results_dir}")
        raise SystemExit(1)

    click.echo(f"Generating charts for {results_dir}...")

    generate_charts(str(results_path))

    click.echo(f"Charts saved to {results_dir}")


@cli.command("check-setup")
def check_setup():
    """Verify installation and API connectivity."""
    import sys

    click.echo("Checking setup...")

    # Check Python version
    click.echo(f"  Python: {sys.version_info.major}.{sys.version_info.minor}")

    # Check required packages
    packages = ["httpx", "datasets", "pandas", "scipy", "tenacity", "click", "yaml"]
    for pkg in packages:
        try:
            __import__(pkg if pkg != "yaml" else "yaml")
            click.echo(f"  {pkg}: OK")
        except ImportError:
            click.echo(f"  {pkg}: MISSING")

    # Check API key
    from backend.config import OPENROUTER_API_KEY

    if OPENROUTER_API_KEY:
        click.echo("  OPENROUTER_API_KEY: Set")
    else:
        click.echo("  OPENROUTER_API_KEY: NOT SET (add to .env)")


@cli.command("cache")
@click.argument("action", type=click.Choice(["stats", "clear"]))
def cache(action: str):
    """Manage Stage 1 response cache."""
    from backend.governance.stage1_cache import get_cache

    cache = get_cache()

    if action == "stats":
        stats = cache.stats()
        click.echo(f"Cache statistics:")
        click.echo(f"  Entries: {stats['total_entries']}")
        click.echo(f"  Size: {stats['size_bytes'] / 1024:.1f} KB")

    elif action == "clear":
        if click.confirm("Clear all cached Stage 1 responses?"):
            count = cache.clear()
            click.echo(f"Cleared {count} cached entries")


@cli.command()
@click.argument("manifest_path", type=click.Path(exists=True))
def manifest(manifest_path: str):
    """Display experiment manifest information."""
    from experiments.manifest import load_manifest

    m = load_manifest(manifest_path)

    click.echo("Experiment Manifest")
    click.echo("=" * 40)
    click.echo(f"Timestamp: {m.get('timestamp', 'Unknown')}")

    git = m.get("git", {})
    click.echo(f"Git SHA: {git.get('sha', 'Unknown')[:8]}...")
    click.echo(f"Git Dirty: {git.get('dirty', 'Unknown')}")

    click.echo("\nConfiguration:")
    config = m.get("config", {})
    for key, value in config.items():
        click.echo(f"  {key}: {value}")

    click.echo("\nPackages:")
    packages = m.get("packages", {})
    for pkg, version in packages.items():
        click.echo(f"  {pkg}: {version}")


if __name__ == "__main__":
    cli()
