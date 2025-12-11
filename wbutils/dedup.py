"""Run deduplication logic."""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional


class DedupStrategy(Enum):
    """Strategy for selecting among duplicate runs."""

    LATEST = "latest"  # Pick the most recently created run
    LONGEST = "longest"  # Pick the run with most logged steps
    BEST = "best"  # Pick based on a metric (requires metric_key)
    CUSTOM = "custom"  # Use a custom selector function


@dataclass
class DedupConfig:
    """Configuration for run deduplication."""

    # Config keys to use for identifying duplicates
    # e.g., ["task_name", "seed", "lr"]
    keys: list[str] = field(default_factory=lambda: ["task_name", "seed"])

    # Strategy for selecting among duplicates
    strategy: DedupStrategy = DedupStrategy.LATEST

    # For BEST strategy: which metric to use and whether higher is better
    metric_key: Optional[str] = None
    metric_higher_is_better: bool = True

    # For CUSTOM strategy: a function (list[Run]) -> Run
    custom_selector: Optional[Callable] = None

    # Whether to include runs with missing keys
    include_incomplete: bool = False


def get_dedup_key(run, keys: list[str]) -> Optional[tuple]:
    """
    Extract deduplication key from a run's config.

    Args:
        run: Wandb run object or CachedRun
        keys: List of config keys to use

    Returns:
        Tuple of values for the keys, or None if any key is missing
    """
    config = run.config if isinstance(run.config, dict) else dict(run.config)
    values = []

    for key in keys:
        # Support nested keys with dot notation (e.g., "model.hidden_size")
        value = config
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None  # Key not found
        values.append(value)

    # Convert lists/dicts to strings for hashability
    hashable_values = []
    for v in values:
        if isinstance(v, (list, dict)):
            hashable_values.append(str(v))
        else:
            hashable_values.append(v)

    return tuple(hashable_values)


def _select_latest(runs: list) -> "Run":
    """Select the most recently created run."""
    return max(runs, key=lambda r: r.created_at)


def _select_longest(runs: list) -> "Run":
    """Select the run with the most logged steps."""

    def get_step_count(run):
        # Get summary - works for both wandb Run and CachedRun
        summary = run.summary if isinstance(run.summary, dict) else dict(run.summary)
        if "_step" in summary:
            return summary["_step"]
        return 0

    return max(runs, key=get_step_count)


def _select_best(runs: list, metric_key: str, higher_is_better: bool = True) -> "Run":
    """Select the run with best value for a metric."""

    def get_metric(run):
        summary = run.summary if isinstance(run.summary, dict) else dict(run.summary)
        if metric_key in summary:
            val = summary[metric_key]
            if val is None:
                return float("-inf") if higher_is_better else float("inf")
            return val
        return float("-inf") if higher_is_better else float("inf")

    if higher_is_better:
        return max(runs, key=get_metric)
    else:
        return min(runs, key=get_metric)


class RunDeduplicator:
    """Handles deduplication of wandb runs."""

    def __init__(self, config: Optional[DedupConfig] = None):
        """
        Initialize deduplicator.

        Args:
            config: Deduplication configuration
        """
        self.config = config or DedupConfig()

    def deduplicate(self, runs: list) -> tuple[list, dict]:
        """
        Deduplicate a list of runs.

        Args:
            runs: List of wandb Run objects

        Returns:
            Tuple of (deduplicated_runs, duplicate_groups)
            - deduplicated_runs: List of selected runs
            - duplicate_groups: Dict mapping dedup_key -> list of all runs with that key
        """
        # Group runs by dedup key
        groups = defaultdict(list)
        incomplete_runs = []

        for run in runs:
            key = get_dedup_key(run, self.config.keys)
            if key is None:
                if self.config.include_incomplete:
                    incomplete_runs.append(run)
            else:
                groups[key].append(run)

        # Select from each group
        selected = []
        for key, group_runs in groups.items():
            chosen = self._select_run(group_runs)
            selected.append(chosen)

        # Add incomplete runs if configured
        selected.extend(incomplete_runs)

        return selected, dict(groups)

    def _select_run(self, runs: list) -> "Run":
        """Select a single run from a group of duplicates."""
        if len(runs) == 1:
            return runs[0]

        strategy = self.config.strategy

        if strategy == DedupStrategy.LATEST:
            return _select_latest(runs)
        elif strategy == DedupStrategy.LONGEST:
            return _select_longest(runs)
        elif strategy == DedupStrategy.BEST:
            if not self.config.metric_key:
                raise ValueError("BEST strategy requires metric_key to be set")
            return _select_best(runs, self.config.metric_key, self.config.metric_higher_is_better)
        elif strategy == DedupStrategy.CUSTOM:
            if not self.config.custom_selector:
                raise ValueError("CUSTOM strategy requires custom_selector to be set")
            return self.config.custom_selector(runs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def get_duplicate_stats(self, groups: dict) -> dict:
        """
        Get statistics about duplicates.

        Args:
            groups: Duplicate groups from deduplicate()

        Returns:
            Dict with statistics
        """
        group_sizes = [len(g) for g in groups.values()]

        return {
            "total_groups": len(groups),
            "total_runs": sum(group_sizes),
            "unique_runs": len([s for s in group_sizes if s == 1]),
            "duplicate_groups": len([s for s in group_sizes if s > 1]),
            "max_duplicates": max(group_sizes) if group_sizes else 0,
            "avg_duplicates": sum(group_sizes) / len(group_sizes) if group_sizes else 0,
        }
