"""Command-line interface for wbutils."""

import argparse
import sys
from pathlib import Path

from .dedup import DedupStrategy
from .discovery import MetricSource
from .inspector import WandbInspector


def parse_filters(filter_args: list[str] | None) -> dict[str, list[str]]:
    """
    Parse filter arguments into a dict.

    Args:
        filter_args: List of "key=value" or "key=val1,val2" strings

    Returns:
        Dict mapping key -> list of acceptable values
    """
    if not filter_args:
        return {}

    filters = {}
    for f in filter_args:
        if "=" not in f:
            print(f"Warning: Invalid filter '{f}' (expected key=value)")
            continue
        key, value = f.split("=", 1)
        # Support comma-separated values for OR
        values = [v.strip() for v in value.split(",")]
        filters[key] = values
    return filters


def get_nested_value(d: dict, key: str):
    """
    Get a value from a nested dict using dot notation.

    Args:
        d: Dictionary to search
        key: Key with optional dot notation (e.g., "agent.tau" or "env_name")

    Returns:
        Value if found, None otherwise
    """
    if not d or not key:
        return None

    # Try direct key first (handles keys that literally contain dots)
    if key in d:
        return d[key]

    # Try nested access
    if "." in key:
        parts = key.split(".")
        current = d
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    return None


def get_all_config_keys(config: dict, prefix: str = "") -> set[str]:
    """
    Get all keys from a config dict, including nested keys with dot notation.

    Args:
        config: Config dictionary
        prefix: Current key prefix for recursion

    Returns:
        Set of all keys (e.g., {"env_name", "agent.tau", "agent.lr"})
    """
    keys = set()
    for k, v in config.items():
        full_key = f"{prefix}.{k}" if prefix else k
        keys.add(full_key)
        if isinstance(v, dict):
            keys.update(get_all_config_keys(v, full_key))
    return keys


def apply_filters(
    runs: list, filters: dict[str, list[str]], show_debug_on_empty: bool = True
) -> list:
    """
    Filter runs by config values.

    Args:
        runs: List of CachedRun objects
        filters: Dict mapping key -> list of acceptable values (AND between keys, OR within values)
                 Keys support dot notation for nested access (e.g., "agent.tau")
        show_debug_on_empty: Show debug info if no runs match

    Returns:
        Filtered list of runs
    """
    if not filters:
        return runs

    filtered = []
    for run in runs:
        config = run.config if isinstance(run.config, dict) else {}
        match = True
        for key, values in filters.items():
            run_value = get_nested_value(config, key)
            # Convert to string for comparison
            run_value_str = str(run_value) if run_value is not None else None
            if run_value_str not in values:
                match = False
                break
        if match:
            filtered.append(run)

    # Debug output if no matches
    if not filtered and show_debug_on_empty and runs:
        print(f"\nWarning: Filter matched 0/{len(runs)} runs")
        print("Filter debug:")
        for key, filter_values in filters.items():
            # Collect actual values for this key
            actual_values = set()
            missing_count = 0
            for run in runs:
                config = run.config if isinstance(run.config, dict) else {}
                val = get_nested_value(config, key)
                if val is not None:
                    actual_values.add(str(val))
                else:
                    missing_count += 1

            if missing_count == len(runs):
                print(f"  {key}: KEY NOT FOUND in any run config")
                # Suggest similar keys (including nested)
                all_keys = set()
                for run in runs:
                    config = run.config if isinstance(run.config, dict) else {}
                    all_keys.update(get_all_config_keys(config))
                similar = [
                    k for k in all_keys if key.lower() in k.lower() or k.lower() in key.lower()
                ]
                if similar:
                    print(f"    Similar keys: {sorted(similar)[:5]}")
            else:
                actual_list = sorted([v for v in actual_values if v is not None])[:10]
                print(f"  {key}: filter={filter_values}")
                print(
                    f"    actual values: {actual_list}"
                    + (" ..." if len(actual_values) > 10 else "")
                )
                # Check for near matches
                for fv in filter_values:
                    near = [
                        av for av in actual_values if av and fv.lower() == av.lower() and fv != av
                    ]
                    if near:
                        print(f"    Did you mean '{near[0]}' instead of '{fv}'?")
        print()

    return filtered


def parse_metric_maps(metric_map_args: list[str] | None) -> dict[str, dict[str, str]]:
    """
    Parse metric mapping arguments.

    Formats:
        LABEL=alt_metric           -> {LABEL: {None: alt_metric}} (single metric mode)
        LABEL:canonical=actual     -> {LABEL: {canonical: actual}} (multi-metric mode)

    Args:
        metric_map_args: List of mapping strings

    Returns:
        Dict mapping label -> {canonical_metric: actual_metric}
    """
    if not metric_map_args:
        return {}

    mappings = {}  # label -> {canonical: actual}

    for m in metric_map_args:
        if ":" in m and "=" in m:
            # Format: LABEL:canonical=actual
            label_part, mapping_part = m.split(":", 1)
            if "=" in mapping_part:
                canonical, actual = mapping_part.split("=", 1)
                if label_part not in mappings:
                    mappings[label_part] = {}
                mappings[label_part][canonical] = actual
        elif "=" in m:
            # Format: LABEL=alt_metric (single metric mode - use None as key)
            label, actual = m.split("=", 1)
            if label not in mappings:
                mappings[label] = {}
            mappings[label][None] = actual
        else:
            print(
                f"Warning: Invalid metric-map format '{m}' (expected LABEL=metric or LABEL:canonical=actual)"
            )

    return mappings


def get_mapped_metric(
    label: str, canonical_metric: str, metric_maps: dict, metrics: list[str]
) -> str:
    """Get the actual metric name for a project, considering mappings."""
    if label not in metric_maps:
        return canonical_metric

    label_maps = metric_maps[label]

    # Check for specific mapping
    if canonical_metric in label_maps:
        return label_maps[canonical_metric]

    # Check for single-metric mode (None key)
    if None in label_maps and len(metrics) == 1:
        return label_maps[None]

    return canonical_metric


def parse_project_specs(
    project_args: list[str] | None, default_entity: str | None
) -> list[tuple[str, str, str]]:
    """
    Parse project specification arguments.

    Formats:
        label:project           -> (label, default_entity, project)
        label:entity/project    -> (label, entity, project)
        project                 -> (project, default_entity, project)
        entity/project          -> (project, entity, project)

    Args:
        project_args: List of project spec strings
        default_entity: Default entity to use when not specified in spec

    Returns:
        List of (label, entity, project) tuples
    """
    if not project_args:
        return []

    results = []

    for p in project_args:
        if ":" in p:
            # Has explicit label
            label, path = p.split(":", 1)
            if "/" in path:
                # label:entity/project
                entity, project = path.split("/", 1)
            else:
                # label:project (use default entity)
                if not default_entity:
                    print(f"Error: No default entity for '{p}'. Use 'label:entity/project' format.")
                    return []
                entity, project = default_entity, path
        else:
            # No label
            if "/" in p:
                # entity/project (label = project name)
                entity, project = p.split("/", 1)
                label = project
            else:
                # Just project name (use default entity, label = project)
                if not default_entity:
                    print(f"Error: No default entity for '{p}'. Use 'entity/project' format.")
                    return []
                entity, project = default_entity, p
                label = project

        results.append((label, entity, project))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Inspect wandb projects: discover metrics, deduplicate runs, generate reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover metrics in a project
  wbutils discover my-entity my-project

  # List metrics with at least 5 runs
  wbutils metrics my-entity my-project --min-runs 5

  # Generate LaTeX table
  wbutils report my-entity my-project --metrics accuracy loss --group-by task_name

  # Export to CSV
  wbutils export my-entity my-project --metrics accuracy --output results.csv
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    def add_common_args(p, multi_project=False):
        p.add_argument("entity", help="Wandb entity (username or team)")
        if multi_project:
            p.add_argument("projects", nargs="+", help="Wandb project name(s)")
        else:
            p.add_argument("project", help="Wandb project name")
        p.add_argument("--cache-dir", help="Directory to cache data (default: .cache/wbutils)")
        p.add_argument("--no-cache", action="store_true", help="Disable caching")
        p.add_argument(
            "--filter",
            action="append",
            dest="filters",
            metavar="KEY=VALUE",
            help="Filter runs by config (e.g., --filter env_name=square). Use comma for OR: key=a,b. Repeat for AND.",
        )
        p.add_argument(
            "--dedup-keys",
            nargs="+",
            default=None,
            help="Config keys for deduplication (e.g., --dedup-keys task_name seed)",
        )
        p.add_argument(
            "--dedup-strategy",
            choices=["latest", "longest", "best"],
            default="latest",
            help="Strategy for selecting among duplicates",
        )

    # Arguments for analysis commands (support cross-entity)
    def add_analysis_args(p):
        """Add arguments for analysis commands that support cross-entity comparison."""
        p.add_argument(
            "entity",
            nargs="?",
            default=None,
            help="Default entity (optional if using entity/project in --project)",
        )
        p.add_argument(
            "project",
            nargs="?",
            default=None,
            help="Wandb project name (optional if using --project)",
        )
        p.add_argument(
            "--project",
            dest="projects",
            action="append",
            metavar="LABEL:PROJECT",
            help="Project with label. Formats: 'label:project' (uses default entity), "
            "'label:entity/project' (explicit entity). Can be repeated.",
        )
        p.add_argument("--cache-dir", help="Directory to cache data (default: .cache/wbutils)")
        p.add_argument("--no-cache", action="store_true", help="Disable caching")
        p.add_argument(
            "--filter",
            action="append",
            dest="filters",
            metavar="KEY=VALUE",
            help="Filter runs by config (e.g., --filter env_name=square). Use comma for OR: key=a,b. Repeat for AND.",
        )
        p.add_argument(
            "--dedup-keys",
            nargs="+",
            default=None,
            help="Config keys for deduplication (e.g., --dedup-keys task_name seed)",
        )
        p.add_argument(
            "--dedup-strategy",
            choices=["latest", "longest", "best"],
            default="latest",
            help="Strategy for selecting among duplicates",
        )

    # Discover command (supports multiple projects)
    discover_parser = subparsers.add_parser("discover", help="Discover metrics in project(s)")
    add_common_args(discover_parser, multi_project=True)
    discover_parser.add_argument(
        "--max-runs", type=int, default=50, help="Max runs to scan per project"
    )
    discover_parser.add_argument("--no-history", action="store_true", help="Skip history metrics")
    discover_parser.add_argument("--refresh", action="store_true", help="Force re-scan all runs")
    discover_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="List discovered metrics")
    add_common_args(metrics_parser)
    metrics_parser.add_argument("--min-runs", type=int, default=1, help="Min runs for metric")
    metrics_parser.add_argument(
        "--source", choices=["summary", "history", "both"], help="Filter by source"
    )
    metrics_parser.add_argument("--pattern", help="Filter by name pattern")
    metrics_parser.add_argument("--names-only", action="store_true", help="Output names only")

    # Configs command
    configs_parser = subparsers.add_parser("configs", help="List config keys across runs")
    add_common_args(configs_parser)
    configs_parser.add_argument(
        "--min-runs", type=int, default=1, help="Min runs that have this key"
    )
    configs_parser.add_argument("--pattern", help="Filter by name pattern")
    configs_parser.add_argument(
        "--show-values", action="store_true", help="Show unique values for each key"
    )
    configs_parser.add_argument("--refresh", action="store_true", help="Force refresh from wandb")

    # Inspect command (NEW) - inspect a specific metric
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a specific metric across runs")
    add_common_args(inspect_parser)
    inspect_parser.add_argument("--metric", required=True, help="Metric name to inspect")
    inspect_parser.add_argument(
        "--max-runs", type=int, default=10, help="Max runs to inspect for history"
    )
    inspect_parser.add_argument("--show-runs", action="store_true", help="Show per-run details")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report")
    add_analysis_args(report_parser)
    report_parser.add_argument("--metrics", nargs="+", required=True, help="Metrics to include")
    report_parser.add_argument("--config-keys", nargs="+", help="Config keys to include in output")
    report_parser.add_argument(
        "--group-by", nargs="+", help="Config key(s) to group by (e.g., --group-by env_name seed)"
    )
    report_parser.add_argument("--no-std", action="store_true", help="Hide standard deviation")
    report_parser.add_argument(
        "--format", choices=["latex", "text"], default="text", help="Output format"
    )
    report_parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    report_parser.add_argument("--caption", default="Results", help="Table caption")
    report_parser.add_argument("--label", default="tab:results", help="LaTeX label")
    report_parser.add_argument("--refresh", action="store_true", help="Force refresh from wandb")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export runs to CSV")
    add_analysis_args(export_parser)
    export_parser.add_argument("--metrics", nargs="+", help="Metrics to include")
    export_parser.add_argument("--config-keys", nargs="+", help="Config keys to include")
    export_parser.add_argument("--output", "-o", required=True, help="Output file")
    export_parser.add_argument("--no-dedup", action="store_true", help="Don't deduplicate runs")
    export_parser.add_argument("--refresh", action="store_true", help="Force refresh from wandb")

    # History command - fetch and cache time series data
    history_parser = subparsers.add_parser(
        "history", help="Fetch and cache metric history (time series)"
    )
    add_common_args(history_parser)
    history_parser.add_argument("--metric", required=True, help="Metric name to fetch")
    history_parser.add_argument(
        "--config-keys", nargs="+", help="Config keys to include when exporting"
    )
    history_parser.add_argument("--output", "-o", help="Export to CSV file (optional)")
    history_parser.add_argument("--max-runs", type=int, help="Max runs to fetch (default: all)")
    history_parser.add_argument(
        "--step-key", default="_step", help="Step column name (default: _step)"
    )
    history_parser.add_argument(
        "--refresh", action="store_true", help="Force re-fetch from wandb (ignore cache)"
    )

    # Plot command - plot cached history data
    plot_parser = subparsers.add_parser("plot", help="Plot cached metric history")
    # Custom args for plot - entity is optional when using full paths in --project
    plot_parser.add_argument(
        "entity",
        nargs="?",
        default=None,
        help="Default entity (optional if using entity/project in --project)",
    )
    plot_parser.add_argument(
        "project", nargs="?", default=None, help="Wandb project name (optional if using --project)"
    )
    plot_parser.add_argument(
        "--project",
        dest="projects",
        action="append",
        metavar="LABEL:PROJECT",
        help="Project with label. Formats: 'label:project' (uses default entity), "
        "'label:entity/project' (explicit entity). Can be repeated.",
    )
    plot_parser.add_argument(
        "--cache-dir", help="Directory to cache data (default: .cache/wbutils)"
    )
    plot_parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    plot_parser.add_argument(
        "--filter",
        action="append",
        dest="filters",
        metavar="KEY=VALUE",
        help="Filter runs by config (e.g., --filter env_name=square). Repeat for AND.",
    )
    plot_parser.add_argument(
        "--dedup-keys", nargs="+", default=None, help="Config keys for deduplication"
    )
    plot_parser.add_argument(
        "--dedup-strategy",
        choices=["latest", "longest", "best"],
        default="latest",
        help="Strategy for selecting among duplicates",
    )
    plot_group = plot_parser.add_mutually_exclusive_group(required=True)
    plot_group.add_argument("--metric", help="Single metric to plot")
    plot_group.add_argument("--metrics", nargs="+", help="Multiple metrics (subplots)")
    plot_parser.add_argument(
        "--metric-map",
        action="append",
        dest="metric_maps",
        metavar="LABEL=METRIC",
        help="Map metric name for a project (e.g., --metric-map EXPO=eval_base/success). "
        "For multi-metric: --metric-map EXPO:eval/success=eval_base/success",
    )
    plot_parser.add_argument("--group-by", nargs="+", help="Config key(s) for grouping lines")
    plot_parser.add_argument("--split-by", help="Config key to split into multiple plots")
    plot_parser.add_argument("--title", help="Plot title (auto-generated if not provided)")
    plot_parser.add_argument("--xlabel", help="X-axis label (default: Step)")
    plot_parser.add_argument("--ylabel", help="Y-axis label (default: from metric name)")
    plot_parser.add_argument("--xlim", nargs=2, type=float, help="X-axis limits (min max)")
    plot_parser.add_argument("--ylim", nargs=2, type=float, help="Y-axis limits (min max)")
    plot_parser.add_argument(
        "--std-bands", action="store_true", default=None, help="Show std deviation bands"
    )
    plot_parser.add_argument(
        "--no-std-bands", action="store_true", help="Disable std deviation bands"
    )
    plot_parser.add_argument(
        "--smooth",
        type=float,
        default=0,
        metavar="ALPHA",
        help="Smoothing factor (0-1). 0=none, 0.9=heavy smoothing (EMA)",
    )
    plot_parser.add_argument(
        "--no-align-zero",
        action="store_true",
        help="Don't align steps to start at zero (default: align to zero)",
    )
    plot_parser.add_argument("--output", "-o", help="Save to file (default: display)")
    plot_parser.add_argument(
        "--figsize", nargs=2, type=float, default=[10, 6], help="Figure size (width height)"
    )
    plot_parser.add_argument("--legend-loc", default="best", help="Legend location (default: best)")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show project statistics")
    add_common_args(stats_parser)
    stats_parser.add_argument("--refresh", action="store_true", help="Force refresh from wandb")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Create inspector
    strategy_map = {
        "latest": DedupStrategy.LATEST,
        "longest": DedupStrategy.LONGEST,
        "best": DedupStrategy.BEST,
    }

    # Execute command
    if args.command == "discover":
        # Discover supports multiple projects
        cmd_discover(args, strategy_map)
    elif args.command == "plot":
        # Plot has special project handling (optional positional + --project flags)
        cmd_plot(args, strategy_map)
    elif args.command == "report":
        # Report supports cross-entity
        cmd_report(args, strategy_map)
    elif args.command == "export":
        # Export supports cross-entity
        cmd_export(args, strategy_map)
    else:
        # Data ingestion commands use single entity/project
        inspector = WandbInspector(
            entity=args.entity,
            project=args.project,
            dedup_keys=args.dedup_keys,
            dedup_strategy=strategy_map[args.dedup_strategy],
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )

        if args.command == "metrics":
            cmd_metrics(inspector, args)
        elif args.command == "configs":
            cmd_configs(inspector, args)
        elif args.command == "inspect":
            cmd_inspect(inspector, args)
        elif args.command == "history":
            cmd_history(inspector, args)
        elif args.command == "stats":
            cmd_stats(inspector, args)


def cmd_discover(args, strategy_map):
    """Run metrics discovery across one or more projects."""
    for project in args.projects:
        print(f"\n{'='*60}")
        print(f"Discovering metrics in {args.entity}/{project}...")
        print("=" * 60)

        inspector = WandbInspector(
            entity=args.entity,
            project=project,
            dedup_keys=args.dedup_keys,
            dedup_strategy=strategy_map[args.dedup_strategy],
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )

        inspector.discover_metrics(
            max_runs=args.max_runs,
            scan_history=not args.no_history,
            verbose=args.verbose,
            refresh=args.refresh,
        )

        metrics = inspector.get_metrics()
        print(f"\nDiscovered {len(metrics)} metrics.")
        inspector.print_metrics()


def cmd_metrics(inspector: WandbInspector, args):
    """List discovered metrics."""
    source = None
    if args.source:
        source = MetricSource(args.source)

    if args.names_only:
        names = inspector.list_metric_names(
            source=source,
            min_run_count=args.min_runs,
        )
        for name in names:
            print(name)
    else:
        inspector.print_metrics(
            source=source,
            min_run_count=args.min_runs,
            pattern=args.pattern,
        )


def cmd_configs(inspector: WandbInspector, args):
    """List config keys across runs."""
    from collections import defaultdict

    runs = inspector.get_runs(refresh=args.refresh)

    # Apply filters
    filters = parse_filters(getattr(args, "filters", None))
    if filters:
        runs = apply_filters(runs, filters)
        print(f"Filtered to {len(runs)} runs")

    if not runs:
        print("No runs found.")
        return

    # Collect config keys and their values
    key_info = defaultdict(lambda: {"count": 0, "values": set()})

    for run in runs:
        config = run.config if isinstance(run.config, dict) else {}
        for key, value in config.items():
            key_info[key]["count"] += 1
            # Only track hashable values
            try:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    key_info[key]["values"].add(value)
                elif isinstance(value, (list, tuple)) and len(value) <= 5:
                    key_info[key]["values"].add(str(value))
            except TypeError:
                pass

    # Filter by min_runs and pattern
    filtered = []
    for key, info in key_info.items():
        if info["count"] < args.min_runs:
            continue
        if args.pattern and args.pattern.lower() not in key.lower():
            continue
        filtered.append((key, info))

    # Sort by count descending, then by name
    filtered.sort(key=lambda x: (-x[1]["count"], x[0]))

    if not filtered:
        print("No config keys found matching criteria.")
        return

    print(f"\n{'Config Key':<40} {'Runs':<8} {'Unique Values'}")
    print("-" * 70)

    for key, info in filtered:
        num_values = len(info["values"])
        if args.show_values and num_values <= 10:
            values_str = str(sorted(info["values"], key=lambda x: (x is None, str(x))))
        else:
            values_str = f"{num_values} unique" if num_values > 0 else "-"
        print(f"{key:<40} {info['count']:<8} {values_str}")


def cmd_inspect(inspector: WandbInspector, args):
    """Inspect a specific metric across runs."""
    import statistics

    import wandb

    metric_name = args.metric
    runs = inspector.get_runs()

    # Apply filters
    filters = parse_filters(getattr(args, "filters", None))
    if filters:
        runs = apply_filters(runs, filters)
        print(f"Filtered to {len(runs)} runs")

    if not runs:
        print("No runs found.")
        return

    print(f"\nInspecting metric: {metric_name}")
    print("=" * 60)

    # Check summary values
    summary_values = []
    runs_with_summary = []
    for run in runs:
        summary = run.summary if isinstance(run.summary, dict) else {}
        if metric_name in summary:
            val = summary[metric_name]
            if isinstance(val, (int, float)):
                summary_values.append(val)
                runs_with_summary.append(run)

    if summary_values:
        print(f"\nSummary (final values):")
        print(f"  Runs with metric: {len(summary_values)}/{len(runs)}")
        print(f"  Mean: {statistics.mean(summary_values):.4f}")
        if len(summary_values) > 1:
            print(f"  Std:  {statistics.stdev(summary_values):.4f}")
        print(f"  Min:  {min(summary_values):.4f}")
        print(f"  Max:  {max(summary_values):.4f}")
    else:
        print(f"\nSummary: Metric not found in run summaries")

    # Check history (requires wandb API call)
    print(f"\nHistory (time series):")
    print(f"  Checking up to {args.max_runs} runs...")

    api = wandb.Api()
    history_info = []

    for i, cached_run in enumerate(runs[: args.max_runs]):
        try:
            # Fetch the actual run from wandb
            run = api.run(f"{inspector.entity}/{inspector.project}/{cached_run.id}")
            history = run.history(keys=[metric_name], pandas=False)

            # Count non-null values
            count = sum(1 for row in history if metric_name in row and row[metric_name] is not None)
            if count > 0:
                history_info.append(
                    {
                        "run_name": cached_run.name,
                        "run_id": cached_run.id,
                        "count": count,
                    }
                )
        except Exception as e:
            if args.show_runs:
                print(f"  Warning: Could not fetch history for {cached_run.id}: {e}")

    if history_info:
        counts = [h["count"] for h in history_info]
        print(f"  Runs with history: {len(history_info)}/{min(len(runs), args.max_runs)}")
        print(f"  Data points per run:")
        print(f"    Mean: {statistics.mean(counts):.1f}")
        print(f"    Min:  {min(counts)}")
        print(f"    Max:  {max(counts)}")

        if args.show_runs:
            print(f"\n  Per-run details:")
            print(f"  {'Run Name':<40} {'Data Points'}")
            print("  " + "-" * 55)
            for h in sorted(history_info, key=lambda x: -x["count"]):
                print(f"  {h['run_name']:<40} {h['count']}")
    else:
        print(f"  Metric not found in history (or not logged as time series)")


def cmd_report(args, strategy_map):
    """Generate report with cross-entity support."""
    from collections import defaultdict

    import numpy as np

    # Parse projects
    project_configs = []  # List of (label, entity, project)

    if args.projects:
        project_configs = parse_project_specs(args.projects, args.entity)
        if not project_configs:
            return  # Error already printed
    elif args.project and args.entity:
        project_configs.append((args.project, args.entity, args.project))
    elif args.entity and not args.project:
        print("Error: Must specify project (positional) or --project flag(s)")
        return
    else:
        print("Error: Must specify project. Use --project flag(s) with entity/project format")
        return

    # Parse filters
    filters = parse_filters(getattr(args, "filters", None))

    # Create inspectors and collect data
    multi_project = len(project_configs) > 1
    group_by = args.group_by or []

    # If multi-project and not explicitly grouping by project, add it
    if multi_project and "_project" not in group_by:
        effective_group_by = ["_project"] + group_by
    else:
        effective_group_by = group_by

    # Collect all runs with their data
    all_data = []  # List of (project_label, run_summary, run_config)

    for label, entity, proj in project_configs:
        inspector = WandbInspector(
            entity=entity,
            project=proj,
            dedup_keys=args.dedup_keys,
            dedup_strategy=strategy_map[args.dedup_strategy],
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )

        runs = inspector.get_runs(config_filters=filters, refresh=args.refresh)

        for run in runs:
            summary = run.summary if isinstance(run.summary, dict) else {}
            config = run.config if isinstance(run.config, dict) else {}
            all_data.append((label, summary, config))

    if not all_data:
        print("No runs found.")
        return

    if filters:
        print(f"Filtered to {len(all_data)} runs")

    # Group and aggregate
    # grouped: group_key -> metric -> [values]
    grouped = defaultdict(lambda: defaultdict(list))

    for label, summary, config in all_data:
        # Build group key
        key_parts = []
        for gk in effective_group_by:
            if gk == "_project":
                key_parts.append(label)
            else:
                key_parts.append(config.get(gk, "unknown"))

        if len(key_parts) == 0:
            group_key = "all"
        elif len(key_parts) == 1:
            group_key = key_parts[0]
        else:
            group_key = tuple(key_parts)

        # Collect metric values
        for m in args.metrics:
            val = summary.get(m)
            if val is not None and isinstance(val, (int, float)):
                grouped[group_key][m].append(val)

    # Compute stats
    class Stats:
        def __init__(self, values):
            self.values = values
            self.count = len(values)
            self.mean = np.mean(values) if values else None
            self.std = np.std(values) if len(values) > 1 else None

    group_stats = {}  # group_key -> metric -> Stats
    for group_key, metrics in grouped.items():
        group_stats[group_key] = {}
        for m, vals in metrics.items():
            group_stats[group_key][m] = Stats(vals)

    # Overall stats
    overall_stats = {}
    for m in args.metrics:
        all_vals = []
        for _, metrics in grouped.items():
            all_vals.extend(metrics.get(m, []))
        overall_stats[m] = Stats(all_vals)

    # Format output
    def fmt_val(stats, show_std: bool) -> str:
        if stats is None or stats.mean is None:
            return "-"
        if show_std and stats.std is not None:
            return f"{stats.mean:.4f} ± {stats.std:.4f}"
        return f"{stats.mean:.4f}"

    show_std = not args.no_std

    if args.format == "latex":
        # Generate LaTeX table
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")

        # Column spec
        n_group_cols = len(effective_group_by) if effective_group_by else 0
        n_metric_cols = len(args.metrics)
        col_spec = "l" * n_group_cols + "r" * n_metric_cols
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")

        # Header
        header_parts = [gk.replace("_", "\\_") for gk in effective_group_by]
        header_parts.extend([m.replace("_", "\\_") for m in args.metrics])
        lines.append(" & ".join(header_parts) + " \\\\")
        lines.append("\\midrule")

        # Data rows
        for group_key in sorted(group_stats.keys(), key=str):
            row_parts = []
            if isinstance(group_key, tuple):
                row_parts.extend([str(v).replace("_", "\\_") for v in group_key])
            elif group_key != "all":
                row_parts.append(str(group_key).replace("_", "\\_"))

            for m in args.metrics:
                stats = group_stats[group_key].get(m)
                row_parts.append(fmt_val(stats, show_std))

            lines.append(" & ".join(row_parts) + " \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{args.caption}}}")
        lines.append(f"\\label{{{args.label}}}")
        lines.append("\\end{table}")

        output = "\n".join(lines)
    else:
        # Text table format
        lines = []
        lines.append(f"Results ({len(all_data)} runs)")
        lines.append("")

        if group_stats and effective_group_by:
            groups = sorted(group_stats.keys(), key=str)

            # Calculate column widths
            col_widths = []
            for i, gk in enumerate(effective_group_by):
                max_width = len(gk)
                for g in groups:
                    val = g[i] if isinstance(g, tuple) else g
                    max_width = max(max_width, len(str(val)))
                col_widths.append(max_width)

            metric_widths = []
            for m in args.metrics:
                max_width = len(m)
                for g in groups:
                    stats = group_stats[g].get(m)
                    val_str = fmt_val(stats, show_std)
                    max_width = max(max_width, len(val_str))
                metric_widths.append(max_width)

            # Header row
            header_parts = []
            for i, gk in enumerate(effective_group_by):
                header_parts.append(gk.ljust(col_widths[i]))
            for i, m in enumerate(args.metrics):
                header_parts.append(m.rjust(metric_widths[i]))

            header = "  ".join(header_parts)
            lines.append(header)
            lines.append("-" * len(header))

            # Data rows
            for g in groups:
                row_parts = []
                if isinstance(g, tuple):
                    for i, val in enumerate(g):
                        row_parts.append(str(val).ljust(col_widths[i]))
                else:
                    row_parts.append(str(g).ljust(col_widths[0]))

                for i, m in enumerate(args.metrics):
                    stats = group_stats[g].get(m)
                    val_str = fmt_val(stats, show_std)
                    row_parts.append(val_str.rjust(metric_widths[i]))

                lines.append("  ".join(row_parts))

            # Summary stats
            lines.append("")
            lines.append("Overall:")
            for m in args.metrics:
                stats = overall_stats.get(m)
                lines.append(f"  {m}: {fmt_val(stats, show_std)} (n={stats.count if stats else 0})")
        else:
            # No grouping - just show overall stats
            for m in args.metrics:
                stats = overall_stats.get(m)
                if stats:
                    lines.append(f"{m}: {fmt_val(stats, show_std)} (n={stats.count})")
                else:
                    lines.append(f"{m}: -")

        output = "\n".join(lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


def cmd_export(args, strategy_map):
    """Export to CSV with cross-entity support."""
    import pandas as pd

    # Parse projects
    project_configs = []  # List of (label, entity, project)

    if args.projects:
        project_configs = parse_project_specs(args.projects, args.entity)
        if not project_configs:
            return  # Error already printed
    elif args.project and args.entity:
        project_configs.append((args.project, args.entity, args.project))
    elif args.entity and not args.project:
        print("Error: Must specify project (positional) or --project flag(s)")
        return
    else:
        print("Error: Must specify project. Use --project flag(s) with entity/project format")
        return

    # Parse filters
    filters = parse_filters(getattr(args, "filters", None))

    multi_project = len(project_configs) > 1

    # Collect all runs
    all_rows = []

    for label, entity, proj in project_configs:
        inspector = WandbInspector(
            entity=entity,
            project=proj,
            dedup_keys=args.dedup_keys,
            dedup_strategy=strategy_map[args.dedup_strategy],
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )

        # Get runs
        if args.no_dedup:
            runs = inspector.get_runs(config_filters=filters, refresh=args.refresh)
        else:
            runs = inspector.get_runs(config_filters=filters, refresh=args.refresh)

        for run in runs:
            summary = run.summary if isinstance(run.summary, dict) else {}
            config = run.config if isinstance(run.config, dict) else {}

            row = {
                "run_id": run.id,
                "run_name": run.name,
            }

            # Add project column if multi-project
            if multi_project:
                row["_project"] = label
                row["_entity"] = entity

            # Add config keys
            if args.config_keys:
                for k in args.config_keys:
                    row[k] = config.get(k)

            # Add metrics
            if args.metrics:
                for m in args.metrics:
                    row[m] = summary.get(m)
            else:
                # Export all summary metrics
                for k, v in summary.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        row[k] = v

            all_rows.append(row)

    if not all_rows:
        print("No runs found.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output, index=False)
    print(f"Exported {len(df)} runs to {args.output}")


def cmd_history(inspector: WandbInspector, args):
    """Fetch and cache metric history (time series) from wandb."""
    import csv
    import json
    from pathlib import Path

    import wandb

    metric_name = args.metric
    step_key = args.step_key
    config_keys = args.config_keys or []
    refresh = getattr(args, "refresh", False)
    output_file = args.output  # None if not specified

    # Parse filters
    filters = parse_filters(getattr(args, "filters", None))

    runs = inspector.get_runs(config_filters=filters)
    if filters:
        print(f"Filtered to {len(runs)} runs")
    if args.max_runs:
        runs = runs[: args.max_runs]

    if not runs:
        print("No runs found.")
        return

    # Setup history cache directory
    cache_dir = None
    if inspector.cache_dir:
        cache_dir = inspector.cache_dir / "history" / f"{inspector.entity}_{inspector.project}"
        cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(run_id: str, metric: str) -> Path | None:
        if not cache_dir:
            return None
        safe_metric = metric.replace("/", "_").replace("\\", "_")
        return cache_dir / f"{run_id}_{safe_metric}.json"

    def load_from_cache(run_id: str, metric: str) -> tuple[list, str | None] | None:
        """Load history from cache. Returns (data, step_key) or None."""
        path = get_cache_path(run_id, metric)
        if path and path.exists() and not refresh:
            try:
                with open(path) as f:
                    cached = json.load(f)
                # Handle both old format (list) and new format (dict with metadata)
                if isinstance(cached, list):
                    return cached, None  # Old format, unknown step_key
                elif isinstance(cached, dict) and "data" in cached:
                    return cached["data"], cached.get("step_key")
            except Exception:
                pass
        return None

    def save_to_cache(run_id: str, metric: str, data: list, step_key: str):
        """Save history to cache with step_key metadata."""
        path = get_cache_path(run_id, metric)
        if path:
            try:
                with open(path, "w") as f:
                    json.dump({"step_key": step_key, "data": data}, f)
            except Exception:
                pass

    print(f"Fetching history for '{metric_name}' from {len(runs)} runs (step_key={step_key})...")

    api = wandb.Api()
    cache_hits = 0
    total_points = 0
    run_stats = []  # For summary
    step_key_mismatch = False

    for i, cached_run in enumerate(runs):
        # Try cache first
        cache_result = load_from_cache(cached_run.id, metric_name)

        if cache_result is not None:
            cached_history, cached_step_key = cache_result

            # Warn if cached with different step_key
            if cached_step_key and cached_step_key != step_key:
                if not step_key_mismatch:
                    print(
                        f"\n  Warning: Cache was created with step_key='{cached_step_key}', "
                        f"but you requested '{step_key}'."
                    )
                    print(f"  Use --refresh to re-fetch with new step_key.\n")
                    step_key_mismatch = True

            count = len(cached_history)
            cache_hits += 1
            total_points += count
            run_stats.append({"run": cached_run, "count": count, "cached": True})
            print(f"  [{i+1}/{len(runs)}] {cached_run.name}: {count} points (cached)")
        else:
            # Fetch from wandb
            try:
                run = api.run(f"{inspector.entity}/{inspector.project}/{cached_run.id}")
                history = run.history(keys=[metric_name, step_key], pandas=False)

                history_data = []
                for row in history:
                    if metric_name in row and row[metric_name] is not None:
                        step = row.get(step_key, len(history_data))
                        value = row[metric_name]
                        history_data.append({"step": step, "value": value})

                count = len(history_data)
                total_points += count
                run_stats.append({"run": cached_run, "count": count, "cached": False})

                # Save to cache with step_key metadata
                if history_data:
                    save_to_cache(cached_run.id, metric_name, history_data, step_key)

                print(f"  [{i+1}/{len(runs)}] {cached_run.name}: {count} points")

            except Exception as e:
                print(f"  [{i+1}/{len(runs)}] {cached_run.name}: ERROR - {e}")
                run_stats.append({"run": cached_run, "count": 0, "cached": False})

    # Summary
    runs_with_data = [s for s in run_stats if s["count"] > 0]
    print(f"\nCached {total_points} data points from {len(runs_with_data)}/{len(runs)} runs")
    if cache_hits > 0:
        print(f"  ({cache_hits} runs already cached)")
    if cache_dir:
        print(f"  Cache: {cache_dir}")

    # Export to CSV if output specified
    if output_file:
        rows = []
        for stat in run_stats:
            cached_run = stat["run"]
            config = cached_run.config if isinstance(cached_run.config, dict) else {}
            config_vals = {k: config.get(k, "") for k in config_keys}

            cache_result = load_from_cache(cached_run.id, metric_name)
            history_data = cache_result[0] if cache_result else []
            for row in history_data:
                rows.append(
                    {
                        "run_id": cached_run.id,
                        "run_name": cached_run.name,
                        **config_vals,
                        "step": row["step"],
                        metric_name: row["value"],
                    }
                )

        if rows:
            fieldnames = ["run_id", "run_name"] + config_keys + ["step", metric_name]
            with open(output_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nExported to {output_file}")


def cmd_plot(args, strategy_map):
    """Plot cached metric history."""
    import matplotlib

    # Use Agg backend if saving to file (works without display)
    if args.output:
        matplotlib.use("Agg")
    from collections import defaultdict
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    # Parse projects - supports cross-entity
    # Formats:
    #   --project label:project (uses default entity)
    #   --project label:entity/project (explicit entity)
    project_configs = []  # List of (label, entity, project)

    if args.projects:
        # Using --project flags
        project_configs = parse_project_specs(args.projects, args.entity)
        if not project_configs:
            return  # Error already printed
    elif args.project and args.entity:
        # Using positional entity + project
        project_configs.append((args.project, args.entity, args.project))
    elif args.entity and not args.project:
        # Just entity provided - error
        print("Error: Must specify project (positional) or --project flag(s)")
        return
    else:
        print("Error: Must specify project. Use --project flag(s) with entity/project format")
        print("Example: wbutils plot --project label:entity/project --metric eval/success")
        return

    # Create inspectors for each project
    inspectors = {}  # label -> inspector
    for label, entity, proj in project_configs:
        inspectors[label] = WandbInspector(
            entity=entity,
            project=proj,
            dedup_keys=args.dedup_keys,
            dedup_strategy=strategy_map[args.dedup_strategy],
            cache_dir=args.cache_dir,
            no_cache=args.no_cache,
        )

    # Get metrics to plot
    metrics = args.metrics if args.metrics else [args.metric]
    group_by = args.group_by or []
    split_by = args.split_by

    # Parse filters
    filters = parse_filters(getattr(args, "filters", None))

    # Parse metric mappings
    metric_maps = parse_metric_maps(getattr(args, "metric_maps", None))

    # Check if history is cached for all projects (using mapped metrics)
    for label, inspector in inspectors.items():
        for metric in metrics:
            actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)
            if not inspector.has_history(actual_metric):
                print(f"Error: No cached history for '{actual_metric}' in project '{label}'")
                print(
                    f"Run first: wbutils history {inspector.entity} {inspector.project} --metric {actual_metric}"
                )
                return

    # Validate step_keys are consistent across projects
    step_keys_by_project = {}  # label -> {metric: step_key}
    for label, inspector in inspectors.items():
        step_keys_by_project[label] = {}
        for metric in metrics:
            actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)
            step_key = inspector.get_history_step_key(actual_metric)
            step_keys_by_project[label][metric] = step_key

    # Check for mismatches
    if len(inspectors) > 1:
        for metric in metrics:
            step_keys = {}
            for label in inspectors:
                sk = step_keys_by_project[label].get(metric)
                if sk:
                    step_keys[label] = sk

            unique_step_keys = set(step_keys.values())
            if len(unique_step_keys) > 1:
                print(f"Error: Projects have different step_keys for '{metric}':")
                for label, sk in step_keys.items():
                    actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)
                    print(f"  {label} ({actual_metric}): step_key='{sk}'")
                print("\nAll projects must use the same step_key for comparable plots.")
                print("Re-cache with consistent --step-key:")
                for label, inspector in inspectors.items():
                    actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)
                    print(
                        f"  wbutils history {inspector.entity} {inspector.project} --metric {actual_metric} --step-key _step --refresh"
                    )
                return

    # Collect all data with project labels
    # Structure: {project_label: {run_id: config}}
    all_run_configs = {}
    all_history = []  # List of (project_label, canonical_metric, history_row)
    total_runs_before_filter = 0
    total_runs_after_filter = 0

    # Track min step per run for normalization
    run_min_steps = {}  # (project_label, run_id) -> min_step

    for label, inspector in inspectors.items():
        # Get all runs first, then apply filters with debug output
        all_runs = inspector.get_runs()
        total_runs_before_filter += len(all_runs)

        if filters:
            runs = apply_filters(all_runs, filters)
        else:
            runs = all_runs
        total_runs_after_filter += len(runs)

        all_run_configs[label] = {r.id: r.config for r in runs}

        # Get run IDs for filtering history
        valid_run_ids = set(r.id for r in runs)

        for metric in metrics:
            # Get actual metric name for this project
            actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)

            for row in inspector.get_history(actual_metric):
                # Only include history for runs that pass the filter
                if row["run_id"] in valid_run_ids:
                    # Track min step per run
                    run_key = (label, row["run_id"])
                    if run_key not in run_min_steps:
                        run_min_steps[run_key] = row["step"]
                    else:
                        run_min_steps[run_key] = min(run_min_steps[run_key], row["step"])

                    # Store with canonical metric name for consistent grouping
                    all_history.append((label, metric, row))

    # Normalize steps to start at zero (unless disabled)
    align_zero = not getattr(args, "no_align_zero", False)
    if align_zero and run_min_steps:
        normalized_history = []
        for label, metric, row in all_history:
            run_key = (label, row["run_id"])
            min_step = run_min_steps.get(run_key, 0)
            normalized_row = {**row, "step": row["step"] - min_step}
            normalized_history.append((label, metric, normalized_row))
        all_history = normalized_history

    if filters and not all_history:
        if total_runs_after_filter == 0:
            print("No data after applying filters.")
        else:
            print(
                f"Warning: {total_runs_after_filter} runs matched filters, but no history data found."
            )
            print("This could mean:")
            print("  - History wasn't cached for the filtered runs")
            print("  - The cached history doesn't overlap with filtered run IDs")
            print("\nTry checking cached history vs run IDs:")
            for label, inspector in inspectors.items():
                for metric in metrics:
                    actual_metric = get_mapped_metric(label, metric, metric_maps, metrics)
                    history = inspector.get_history(actual_metric)
                    history_run_ids = set(r["run_id"] for r in history)
                    filtered_run_ids = set(all_run_configs[label].keys())
                    overlap = history_run_ids & filtered_run_ids
                    print(
                        f"  {label}: {len(history_run_ids)} history runs, {len(filtered_run_ids)} filtered runs, {len(overlap)} overlap"
                    )
        return

    if filters and total_runs_after_filter > 0:
        print(f"Filtered to {total_runs_after_filter}/{total_runs_before_filter} runs")

    # Determine split values if split_by is set
    split_values = set()
    if split_by:
        for label, metric, row in all_history:
            run_id = row["run_id"]
            config = all_run_configs[label].get(run_id, {})
            split_val = config.get(split_by, "unknown")
            split_values.add(split_val)
        split_values = sorted(split_values, key=str)
    else:
        split_values = [None]  # Single plot

    # Determine if we should show std bands
    show_std = args.std_bands
    if args.no_std_bands:
        show_std = False

    multi_project = len(inspectors) > 1

    # Create plots
    for split_idx, split_val in enumerate(split_values):
        # Filter data for this split
        if split_by:
            filtered_history = []
            for label, metric, row in all_history:
                run_id = row["run_id"]
                config = all_run_configs[label].get(run_id, {})
                if config.get(split_by) == split_val:
                    filtered_history.append((label, metric, row))
        else:
            filtered_history = all_history

        # Create figure for this split
        fig, axes = plt.subplots(
            len(metrics),
            1,
            figsize=(args.figsize[0], args.figsize[1] * len(metrics)),
            squeeze=False,
        )

        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx, 0]

            # Filter to this metric
            metric_history = [(l, r) for l, m, r in filtered_history if m == metric]

            if not metric_history:
                ax.text(
                    0.5,
                    0.5,
                    f"No data for {metric}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            # Group data: group_key -> step -> [values]
            grouped = defaultdict(lambda: defaultdict(list))

            for proj_label, row in metric_history:
                run_id = row["run_id"]
                step = row["step"]
                value = row["value"]

                # Build group key
                config = all_run_configs[proj_label].get(run_id, {})

                key_parts = []
                if multi_project:
                    key_parts.append(("_project", proj_label))
                for k in group_by:
                    key_parts.append((k, config.get(k, "unknown")))

                if key_parts:
                    group_key = tuple(v for _, v in key_parts)
                    if len(group_key) == 1:
                        group_key = group_key[0]
                else:
                    group_key = "all"

                grouped[group_key][step].append(value)

            # Check if std bands make sense
            has_multiple = any(
                any(len(vals) > 1 for vals in steps.values()) for steps in grouped.values()
            )

            if show_std is None:
                show_std_for_plot = has_multiple
            else:
                show_std_for_plot = show_std

            # Plot each group
            for group_key in sorted(grouped.keys(), key=str):
                steps_data = grouped[group_key]
                steps = sorted(steps_data.keys())

                means = []
                stds = []
                for s in steps:
                    vals = steps_data[s]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals) if len(vals) > 1 else 0)

                steps = np.array(steps)
                means = np.array(means)
                stds = np.array(stds)

                # Apply smoothing (EMA) if requested
                smooth_alpha = getattr(args, "smooth", 0)
                if smooth_alpha > 0 and len(means) > 1:
                    smoothed_means = np.zeros_like(means)
                    smoothed_means[0] = means[0]
                    for i in range(1, len(means)):
                        smoothed_means[i] = (
                            smooth_alpha * smoothed_means[i - 1] + (1 - smooth_alpha) * means[i]
                        )
                    means = smoothed_means

                    # Also smooth stds for consistent bands
                    if np.any(stds > 0):
                        smoothed_stds = np.zeros_like(stds)
                        smoothed_stds[0] = stds[0]
                        for i in range(1, len(stds)):
                            smoothed_stds[i] = (
                                smooth_alpha * smoothed_stds[i - 1] + (1 - smooth_alpha) * stds[i]
                            )
                        stds = smoothed_stds

                # Format label
                if group_key == "all":
                    label = metric if len(metrics) == 1 and not multi_project else None
                elif isinstance(group_key, tuple):
                    # Build label from group key parts
                    label_parts = []
                    key_names = (["_project"] if multi_project else []) + group_by
                    for name, val in zip(key_names, group_key):
                        if name == "_project":
                            label_parts.append(str(val))
                        else:
                            label_parts.append(f"{val}")
                    label = "/".join(label_parts)
                else:
                    label = str(group_key)

                # Plot line
                (line,) = ax.plot(steps, means, label=label)

                # Plot std bands
                if show_std_for_plot and np.any(stds > 0):
                    ax.fill_between(
                        steps, means - stds, means + stds, alpha=0.2, color=line.get_color()
                    )

            # Labels and formatting
            ax.set_xlabel(args.xlabel or "Step")
            ax.set_ylabel(args.ylabel or metric)

            if args.xlim:
                ax.set_xlim(args.xlim)
            if args.ylim:
                ax.set_ylim(args.ylim)

            # Show legend if we have groups
            if group_by or multi_project or len(grouped) > 1:
                ax.legend(loc=args.legend_loc)

            ax.grid(True, alpha=0.3)

        # Title
        if args.title:
            title = args.title
            if split_by and split_val is not None:
                title = f"{title} ({split_by}={split_val})"
        elif split_by and split_val is not None:
            title = f"{split_by}={split_val}"
        elif len(metrics) == 1:
            title = metrics[0]
        else:
            title = None

        if title:
            fig.suptitle(title)

        plt.tight_layout()

        # Output
        if args.output:
            if split_by and len(split_values) > 1:
                # Add suffix for split plots
                p = Path(args.output)
                safe_val = str(split_val).replace("/", "_").replace(" ", "_")
                output_path = p.parent / f"{p.stem}_{safe_val}{p.suffix}"
            else:
                output_path = args.output

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Saved to {output_path}")
        else:
            plt.show()

        plt.close(fig)


def cmd_stats(inspector: WandbInspector, args):
    """Show project statistics."""
    runs = inspector.fetch_runs(refresh=args.refresh)
    deduped = inspector.get_runs(deduplicate=True, refresh=args.refresh)
    stats = inspector.get_duplicate_stats()

    print(f"Project: {inspector.entity}/{inspector.project}")
    print("=" * 60)
    print(f"Total runs:        {len(runs)}")
    print(f"After dedup:       {len(deduped)}")
    print(f"Duplicate groups:  {stats['duplicate_groups']}")
    print(f"Max duplicates:    {stats['max_duplicates']}")
    print(f"Dedup keys:        {inspector.dedup_config.keys}")
    print(f"Dedup strategy:    {inspector.dedup_config.strategy.value}")


if __name__ == "__main__":
    main()
