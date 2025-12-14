"""Main WandbInspector class that ties everything together."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import wandb

from .dedup import DedupConfig, DedupStrategy, RunDeduplicator
from .discovery import MetricsDiscovery, MetricSource, MetricsRegistry
from .report import AggregatedReport, LaTeXFormatter, ReportGenerator


@dataclass
class CachedRun:
    """Cached run data - stores everything needed for metrics/reports without wandb API."""

    id: str
    name: str
    config: dict
    summary: dict
    created_at: str
    state: str
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_wandb_run(cls, run) -> "CachedRun":
        """Create CachedRun from a wandb Run object."""
        # Safely extract config
        try:
            raw_config = {}
            for k in run.config.keys():
                try:
                    raw_config[k] = run.config[k]
                except Exception:
                    pass
        except Exception:
            raw_config = {}

        # Safely extract summary - filter to scalars only
        summary = {}
        try:
            for k in run.summary.keys():
                try:
                    v = run.summary[k]
                    # Skip media objects and complex types
                    if isinstance(v, dict):
                        continue
                    if isinstance(v, list):
                        continue
                    if v is not None and not isinstance(v, (int, float, str, bool)):
                        continue
                    summary[k] = v
                except Exception:
                    pass
        except Exception:
            pass

        return cls(
            id=run.id,
            name=run.name,
            config=raw_config,
            summary=summary,
            created_at=run.created_at,
            state=run.state,
            tags=list(run.tags) if run.tags else [],
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CachedRun":
        return cls(**d)


class WandbInspector:
    """
    Main interface for inspecting wandb projects.

    Usage:
        inspector = WandbInspector("my-entity", "my-project")

        # Discover metrics
        inspector.discover_metrics()
        inspector.print_metrics()

        # Get deduplicated runs
        runs = inspector.get_runs(deduplicate=True)

        # Generate reports
        report = inspector.aggregate(["accuracy", "loss"], group_by="task_name")
        latex = inspector.to_latex(report)
    """

    def __init__(
        self,
        entity: str,
        project: str,
        dedup_keys: Optional[list[str]] = None,
        dedup_strategy: DedupStrategy = DedupStrategy.LATEST,
        cache_dir: Optional[str | Path] = None,
        no_cache: bool = False,
    ):
        """
        Initialize the inspector.

        Args:
            entity: Wandb entity (username or team)
            project: Wandb project name
            dedup_keys: Config keys for deduplication (default: None = no dedup)
            dedup_strategy: Strategy for selecting among duplicates
            cache_dir: Directory to cache data (default: .cache/wbutils in cwd)
            no_cache: Disable caching entirely
        """
        self.entity = entity
        self.project = project
        self.api = wandb.Api()
        self._path = f"{entity}/{project}"

        # Setup deduplication - only if keys are provided
        self.dedup_config = DedupConfig(
            keys=dedup_keys or [],
            strategy=dedup_strategy,
        )
        self.deduplicator = RunDeduplicator(self.dedup_config)

        # Setup caching - default to .cache/wbutils in current working directory
        if no_cache:
            self.cache_dir = None
        elif cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.cwd() / ".cache" / "wbutils"

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load or create registry
        self.registry = MetricsRegistry()
        self._load_registry()

        # State - now uses CachedRun objects
        self._runs: Optional[list[CachedRun]] = None
        self._deduped_runs: Optional[list[CachedRun]] = None
        self._duplicate_groups: Optional[dict] = None

        # Load cached runs
        self._load_runs()

        # Formatters
        self.latex = LaTeXFormatter()

    def _registry_path(self) -> Optional[Path]:
        if self.cache_dir:
            return self.cache_dir / f"{self.entity}_{self.project}_metrics.json"
        return None

    def _runs_cache_path(self) -> Optional[Path]:
        if self.cache_dir:
            return self.cache_dir / f"{self.entity}_{self.project}_runs.json"
        return None

    def _load_registry(self):
        """Load registry from cache if available."""
        path = self._registry_path()
        if path and path.exists():
            self.registry.load(path)

    def _save_registry(self):
        """Save registry to cache."""
        path = self._registry_path()
        if path:
            self.registry.save(path)

    def _load_runs(self):
        """Load runs from cache if available."""
        path = self._runs_cache_path()
        if path and path.exists():
            with open(path) as f:
                data = json.load(f)
            self._runs = [CachedRun.from_dict(r) for r in data.get("runs", [])]

    def _save_runs(self):
        """Save runs to cache."""
        path = self._runs_cache_path()
        if path and self._runs:
            data = {"runs": [r.to_dict() for r in self._runs]}
            with open(path, "w") as f:
                json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # Run fetching
    # -------------------------------------------------------------------------

    def fetch_runs(
        self,
        filters: Optional[dict] = None,
        refresh: bool = False,
    ) -> list[CachedRun]:
        """
        Fetch runs from the project.

        Args:
            filters: Wandb filters (e.g., {"state": "finished"}) - only used when refresh=True
            refresh: Force refresh from wandb API

        Returns:
            List of CachedRun objects
        """
        # Return cached runs if available and not refreshing
        if self._runs is not None and not refresh:
            return self._runs

        # Fetch from wandb API with per_page limit
        wandb_runs = self.api.runs(
            self._path,
            filters=filters,
            order="-created_at",
            per_page=50,
        )

        # Convert to CachedRun objects one at a time
        cached_runs = []
        for run in wandb_runs:
            try:
                cached_runs.append(CachedRun.from_wandb_run(run))
            except RecursionError:
                print(
                    f"Warning: Recursion error caching run {getattr(run, 'id', 'unknown')}, skipping"
                )
                continue
            except Exception as e:
                print(f"Warning: Failed to cache run {getattr(run, 'id', 'unknown')}: {e}")
                continue

        self._runs = cached_runs
        self._deduped_runs = None
        self._duplicate_groups = None

        # Save to cache
        self._save_runs()

        return self._runs

    def get_runs(
        self,
        filters: Optional[dict] = None,
        config_filters: Optional[dict[str, list[str]]] = None,
        deduplicate: bool = True,
        refresh: bool = False,
    ) -> list[CachedRun]:
        """
        Get runs, optionally filtered and deduplicated.

        Args:
            filters: Wandb API filters (only used when refresh=True)
            config_filters: Filter runs by config values. Dict mapping key -> list of acceptable values.
                           Multiple keys are AND, multiple values per key are OR.
            deduplicate: Whether to deduplicate runs
            refresh: Force refresh from wandb API

        Returns:
            List of CachedRun objects
        """
        runs = self.fetch_runs(filters=filters, refresh=refresh)

        # Apply config filters
        if config_filters:
            filtered = []
            for run in runs:
                config = run.config if isinstance(run.config, dict) else {}
                match = True
                for key, values in config_filters.items():
                    run_value = config.get(key)
                    run_value_str = str(run_value) if run_value is not None else None
                    if run_value_str not in values:
                        match = False
                        break
                if match:
                    filtered.append(run)
            runs = filtered

        # Skip deduplication if not requested or no keys configured
        if not deduplicate or not self.dedup_config.keys:
            return runs

        # When config_filters are applied, always re-deduplicate
        if config_filters:
            deduped, _ = self.deduplicator.deduplicate(runs)
            return deduped if deduped else runs

        if self._deduped_runs is None:
            self._deduped_runs, self._duplicate_groups = self.deduplicator.deduplicate(runs)

            # Warn if dedup filtered out all runs
            if len(runs) > 0 and len(self._deduped_runs) == 0:
                print(f"Warning: All {len(runs)} runs were filtered out during deduplication.")
                print(f"  Dedup keys: {self.dedup_config.keys}")
                print(f"  These keys may not exist in your run configs.")
                print(f"  Use --dedup-keys to specify correct keys, or check your config.")
                # Fall back to returning all runs
                self._deduped_runs = runs

        return self._deduped_runs

    def get_duplicate_stats(self) -> dict:
        """Get statistics about duplicate runs."""
        if self._duplicate_groups is None:
            self.get_runs(deduplicate=True)
        return self.deduplicator.get_duplicate_stats(self._duplicate_groups)

    # -------------------------------------------------------------------------
    # Metrics discovery
    # -------------------------------------------------------------------------

    def discover_metrics(
        self,
        max_runs: Optional[int] = 50,
        scan_summary: bool = True,
        scan_history: bool = True,
        verbose: bool = False,
        refresh: bool = False,
    ) -> MetricsRegistry:
        """
        Discover metrics from project runs.

        Args:
            max_runs: Maximum runs to scan (None = all)
            scan_summary: Scan summary metrics
            scan_history: Scan history (time-series) metrics (requires refresh=True for fresh data)
            verbose: Print progress
            refresh: Force re-fetch runs from wandb and re-scan

        Returns:
            MetricsRegistry with discovered metrics
        """
        if refresh:
            self.registry = MetricsRegistry()

        runs = self.fetch_runs(refresh=refresh)

        # History scanning only works with fresh wandb data, not cached
        # When using cache, we can only discover from summary
        actual_scan_history = scan_history and refresh
        if scan_history and not refresh and verbose:
            print(
                "Note: History scanning skipped (using cached data). Use refresh=True for history metrics."
            )

        discovery = MetricsDiscovery(registry=self.registry)
        discovery.discover_from_runs(
            runs,
            scan_summary=scan_summary,
            scan_history=actual_scan_history,
            max_runs=max_runs,
            verbose=verbose,
        )

        self._save_registry()
        return self.registry

    def get_metrics(
        self,
        source: Optional[MetricSource] = None,
        min_run_count: int = 1,
        pattern: Optional[str] = None,
    ) -> list:
        """
        Get discovered metrics matching criteria.

        Args:
            source: Filter by source (SUMMARY, HISTORY, BOTH)
            min_run_count: Minimum runs that have this metric
            pattern: Substring to match in metric name

        Returns:
            List of MetricInfo objects
        """
        return self.registry.get_metrics(
            source=source,
            min_run_count=min_run_count,
            name_pattern=pattern,
        )

    def print_metrics(
        self,
        source: Optional[MetricSource] = None,
        min_run_count: int = 1,
        pattern: Optional[str] = None,
    ):
        """Print discovered metrics in a formatted table."""
        metrics = self.get_metrics(source, min_run_count, pattern)

        if not metrics:
            print("No metrics found matching criteria.")
            return

        print(f"\n{'Metric':<40} {'Source':<10} {'Type':<8} {'Runs':<6} {'Time Series'}")
        print("-" * 80)

        for m in metrics:
            ts = "Yes" if m.is_time_series else "No"
            print(f"{m.name:<40} {m.source.value:<10} {m.dtype:<8} {m.run_count:<6} {ts}")

    def list_metric_names(
        self,
        source: Optional[MetricSource] = None,
        min_run_count: int = 1,
    ) -> list[str]:
        """Get list of metric names."""
        return [m.name for m in self.get_metrics(source, min_run_count)]

    # -------------------------------------------------------------------------
    # Report generation
    # -------------------------------------------------------------------------

    def aggregate(
        self,
        metric_keys: list[str],
        config_keys: Optional[list[str]] = None,
        group_by: Optional[str] = None,
        config_filters: Optional[dict[str, list[str]]] = None,
        deduplicate: bool = True,
        refresh: bool = False,
    ) -> AggregatedReport:
        """
        Generate aggregated statistics.

        Args:
            metric_keys: Metrics to aggregate
            config_keys: Config keys to track
            group_by: Config key to group by
            config_filters: Filter runs by config values
            deduplicate: Use deduplicated runs
            refresh: Force refresh from wandb API

        Returns:
            AggregatedReport object
        """
        runs = self.get_runs(
            config_filters=config_filters, deduplicate=deduplicate, refresh=refresh
        )
        generator = ReportGenerator(runs)
        return generator.generate_aggregated_report(
            metric_keys=metric_keys,
            config_keys=config_keys,
            group_by=group_by,
        )

    def run_reports(
        self,
        metric_keys: Optional[list[str]] = None,
        config_keys: Optional[list[str]] = None,
        config_filters: Optional[dict[str, list[str]]] = None,
        deduplicate: bool = True,
        refresh: bool = False,
    ) -> list:
        """
        Generate individual run reports.

        Args:
            metric_keys: Metrics to include
            config_keys: Config keys to include
            config_filters: Filter runs by config values
            deduplicate: Use deduplicated runs
            refresh: Force refresh from wandb API

        Returns:
            List of RunReport objects
        """
        runs = self.get_runs(
            config_filters=config_filters, deduplicate=deduplicate, refresh=refresh
        )
        generator = ReportGenerator(runs)
        return generator.generate_run_reports(
            metric_keys=metric_keys,
            config_keys=config_keys,
        )

    # -------------------------------------------------------------------------
    # LaTeX output
    # -------------------------------------------------------------------------

    def to_latex_aggregated(
        self,
        report: AggregatedReport,
        metric_keys: Optional[list[str]] = None,
        show_std: bool = True,
        caption: str = "Results",
        label: str = "tab:results",
    ) -> str:
        """Generate LaTeX table from aggregated report."""
        if report.groups:
            return self.latex.grouped_table(
                report,
                metric_keys=metric_keys,
                show_std=show_std,
                caption=caption,
                label=label,
                transpose=True,
            )
        else:
            return self.latex.aggregated_table(
                report,
                metric_keys=metric_keys,
                show_std=show_std,
                caption=caption,
                label=label,
            )

    def to_latex_runs(
        self,
        metric_keys: list[str],
        config_keys: Optional[list[str]] = None,
        caption: str = "Run Results",
        label: str = "tab:runs",
        deduplicate: bool = True,
    ) -> str:
        """Generate LaTeX table of individual runs."""
        reports = self.run_reports(
            metric_keys=metric_keys,
            config_keys=config_keys,
            deduplicate=deduplicate,
        )
        return self.latex.runs_table(
            reports,
            metric_keys=metric_keys,
            config_keys=config_keys,
            caption=caption,
            label=label,
        )

    # -------------------------------------------------------------------------
    # Comparison across projects/experiments
    # -------------------------------------------------------------------------

    @classmethod
    def compare(
        cls,
        inspectors: dict[str, "WandbInspector"],
        metric_keys: list[str],
        show_std: bool = True,
        highlight_best: bool = True,
        higher_is_better: Optional[dict[str, bool]] = None,
        caption: str = "Method Comparison",
        label: str = "tab:comparison",
    ) -> str:
        """
        Compare results across multiple inspectors.

        Args:
            inspectors: Dict mapping name to WandbInspector
            metric_keys: Metrics to compare
            show_std: Show standard deviation
            highlight_best: Bold best values
            higher_is_better: Dict mapping metric to whether higher is better
            caption: Table caption
            label: LaTeX label

        Returns:
            LaTeX comparison table
        """
        reports = {}
        for name, inspector in inspectors.items():
            reports[name] = inspector.aggregate(metric_keys)

        formatter = LaTeXFormatter()
        return formatter.comparison_table(
            reports,
            metric_keys=metric_keys,
            show_std=show_std,
            highlight_best=highlight_best,
            higher_is_better=higher_is_better,
            caption=caption,
            label=label,
        )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def set_dedup_keys(self, keys: list[str]):
        """Update deduplication keys."""
        self.dedup_config.keys = keys
        self._deduped_runs = None
        self._duplicate_groups = None

    def set_dedup_strategy(
        self,
        strategy: DedupStrategy,
        metric_key: Optional[str] = None,
        higher_is_better: bool = True,
    ):
        """
        Update deduplication strategy.

        Args:
            strategy: Dedup strategy
            metric_key: Required for BEST strategy
            higher_is_better: For BEST strategy
        """
        self.dedup_config.strategy = strategy
        if strategy == DedupStrategy.BEST:
            self.dedup_config.metric_key = metric_key
            self.dedup_config.metric_higher_is_better = higher_is_better

        self._deduped_runs = None
        self._duplicate_groups = None

    # -------------------------------------------------------------------------
    # Export/utility
    # -------------------------------------------------------------------------

    def list_config_keys(self, min_run_count: int = 1) -> dict[str, int]:
        """
        List all config keys across runs with their counts.

        Args:
            min_run_count: Minimum runs that have this key

        Returns:
            Dict mapping config key to count of runs that have it
        """
        from collections import Counter

        runs = self.get_runs()
        key_counts = Counter()

        for run in runs:
            config = run.config if isinstance(run.config, dict) else {}
            key_counts.update(config.keys())

        return {k: v for k, v in key_counts.items() if v >= min_run_count}

    def get_config_values(self, key: str) -> set:
        """
        Get unique values for a config key across all runs.

        Args:
            key: Config key to check

        Returns:
            Set of unique values
        """
        runs = self.get_runs()
        values = set()

        for run in runs:
            config = run.config if isinstance(run.config, dict) else {}
            if key in config:
                val = config[key]
                try:
                    if isinstance(val, (list, dict)):
                        values.add(str(val))
                    else:
                        values.add(val)
                except TypeError:
                    values.add(str(val))

        return values

    def export_metrics_list(self, path: str | Path, min_run_count: int = 1):
        """Export metric names to a text file."""
        metrics = self.list_metric_names(min_run_count=min_run_count)
        with open(path, "w") as f:
            f.write("\n".join(metrics))

    def to_dataframe(
        self,
        metric_keys: Optional[list[str]] = None,
        config_keys: Optional[list[str]] = None,
        config_filters: Optional[dict[str, list[str]]] = None,
        deduplicate: bool = True,
        refresh: bool = False,
    ):
        """
        Export runs to a pandas DataFrame.

        Args:
            metric_keys: Metrics to include
            config_keys: Config keys to include
            config_filters: Filter runs by config values
            deduplicate: Use deduplicated runs
            refresh: Force refresh from wandb API

        Returns:
            pandas DataFrame
        """
        import pandas as pd

        reports = self.run_reports(
            metric_keys=metric_keys,
            config_keys=config_keys,
            config_filters=config_filters,
            deduplicate=deduplicate,
            refresh=refresh,
        )

        rows = []
        for r in reports:
            row = {
                "run_id": r.run_id,
                "run_name": r.run_name,
                "state": r.state,
                "created_at": r.created_at,
                "step_count": r.step_count,
            }
            row.update({f"config/{k}": v for k, v in r.config.items()})
            row.update(r.metrics)
            rows.append(row)

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # History (time series) access
    # -------------------------------------------------------------------------

    def _history_cache_dir(self) -> Optional[Path]:
        """Get history cache directory."""
        if not self.cache_dir:
            return None
        return self.cache_dir / "history" / f"{self.entity}_{self.project}"

    def _history_cache_path(self, run_id: str, metric: str) -> Optional[Path]:
        """Get cache path for a specific run+metric history."""
        cache_dir = self._history_cache_dir()
        if not cache_dir:
            return None
        safe_metric = metric.replace("/", "_").replace("\\", "_")
        return cache_dir / f"{run_id}_{safe_metric}.json"

    def get_history(self, metric: str, run_id: Optional[str] = None) -> list[dict]:
        """
        Get cached history data for a metric.

        Args:
            metric: Metric name
            run_id: Specific run ID (None = all runs)

        Returns:
            List of dicts with run_id, run_name, step, value
        """
        import json

        runs = self.get_runs()
        if run_id:
            runs = [r for r in runs if r.id == run_id]

        results = []
        for run in runs:
            path = self._history_cache_path(run.id, metric)
            if path and path.exists():
                try:
                    with open(path) as f:
                        cached = json.load(f)
                    # Handle both old format (list) and new format (dict with metadata)
                    if isinstance(cached, list):
                        history = cached
                    elif isinstance(cached, dict) and "data" in cached:
                        history = cached["data"]
                    else:
                        continue

                    for row in history:
                        results.append(
                            {
                                "run_id": run.id,
                                "run_name": run.name,
                                "step": row["step"],
                                "value": row["value"],
                            }
                        )
                except Exception:
                    pass

        return results

    def get_history_step_key(self, metric: str) -> str | None:
        """
        Get the step_key used when caching history for a metric.

        Returns the step_key from the first cached run found, or None if unknown.
        """
        import json

        runs = self.get_runs()
        for run in runs:
            path = self._history_cache_path(run.id, metric)
            if path and path.exists():
                try:
                    with open(path) as f:
                        cached = json.load(f)
                    if isinstance(cached, dict) and "step_key" in cached:
                        return cached["step_key"]
                except Exception:
                    pass
        return None

    def has_history(self, metric: str) -> bool:
        """Check if history is cached for a metric."""
        runs = self.get_runs()
        for run in runs:
            path = self._history_cache_path(run.id, metric)
            if path and path.exists():
                return True
        return False

    def list_cached_history_metrics(self) -> list[str]:
        """List metrics that have cached history data."""
        cache_dir = self._history_cache_dir()
        if not cache_dir or not cache_dir.exists():
            return []

        metrics = set()
        for f in cache_dir.glob("*.json"):
            # Filename format: {run_id}_{metric}.json
            parts = f.stem.split("_", 1)
            if len(parts) == 2:
                metrics.add(parts[1].replace("_", "/"))

        return sorted(metrics)

    def __repr__(self) -> str:
        return f"WandbInspector(entity='{self.entity}', project='{self.project}')"
