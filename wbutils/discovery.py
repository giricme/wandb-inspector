"""Metrics discovery and registry."""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict


class MetricSource(Enum):
    """Where the metric was found."""
    SUMMARY = "summary"   # In run.summary
    HISTORY = "history"   # In run.history (time-series)
    BOTH = "both"


@dataclass
class MetricInfo:
    """Information about a discovered metric."""
    name: str
    source: MetricSource
    dtype: str                          # Inferred data type
    run_count: int = 0                  # How many runs have this metric
    sample_values: list = field(default_factory=list)  # Sample values
    first_seen: Optional[str] = None    # ISO timestamp
    last_seen: Optional[str] = None     # ISO timestamp
    
    # For history metrics
    is_time_series: bool = False
    typical_length: Optional[int] = None  # Typical number of logged points
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> "MetricInfo":
        d = d.copy()
        d["source"] = MetricSource(d["source"])
        return cls(**d)


class MetricsRegistry:
    """Registry for discovered metrics."""
    
    def __init__(self):
        self.metrics: dict[str, MetricInfo] = {}
        self.last_discovery: Optional[datetime] = None
        self._discovery_run_ids: set[str] = set()  # Track which runs we've scanned
    
    def add_metric(
        self, 
        name: str, 
        source: MetricSource,
        dtype: str,
        sample_value: Any = None,
        is_time_series: bool = False,
        typical_length: Optional[int] = None,
    ):
        """Add or update a metric in the registry."""
        now = datetime.now().isoformat()
        
        if name in self.metrics:
            # Update existing
            metric = self.metrics[name]
            metric.run_count += 1
            metric.last_seen = now
            
            # Update source if we found it in a new place
            if metric.source != source and source != metric.source:
                metric.source = MetricSource.BOTH
            
            # Add sample value
            if sample_value is not None and len(metric.sample_values) < 5:
                metric.sample_values.append(sample_value)
                
            # Update time series info
            if is_time_series:
                metric.is_time_series = True
            if typical_length is not None:
                if metric.typical_length is None:
                    metric.typical_length = typical_length
                else:
                    # Running average
                    metric.typical_length = (metric.typical_length + typical_length) // 2
        else:
            # Create new
            self.metrics[name] = MetricInfo(
                name=name,
                source=source,
                dtype=dtype,
                run_count=1,
                sample_values=[sample_value] if sample_value is not None else [],
                first_seen=now,
                last_seen=now,
                is_time_series=is_time_series,
                typical_length=typical_length,
            )
    
    def get_metrics(
        self, 
        source: Optional[MetricSource] = None,
        min_run_count: int = 1,
        name_pattern: Optional[str] = None,
    ) -> list[MetricInfo]:
        """
        Get metrics matching criteria.
        
        Args:
            source: Filter by source (summary/history/both)
            min_run_count: Minimum number of runs that have this metric
            name_pattern: Substring to match in metric name
            
        Returns:
            List of matching MetricInfo objects
        """
        results = []
        for metric in self.metrics.values():
            if source is not None:
                if source == MetricSource.BOTH:
                    if metric.source != MetricSource.BOTH:
                        continue
                elif metric.source not in (source, MetricSource.BOTH):
                    continue
            
            if metric.run_count < min_run_count:
                continue
            
            if name_pattern and name_pattern.lower() not in metric.name.lower():
                continue
            
            results.append(metric)
        
        return sorted(results, key=lambda m: (-m.run_count, m.name))
    
    def save(self, path: str | Path):
        """Save registry to JSON file."""
        path = Path(path)
        data = {
            "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
            "discovery_run_ids": list(self._discovery_run_ids),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self, path: str | Path):
        """Load registry from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        
        self.last_discovery = (
            datetime.fromisoformat(data["last_discovery"]) 
            if data["last_discovery"] else None
        )
        self._discovery_run_ids = set(data.get("discovery_run_ids", []))
        self.metrics = {
            k: MetricInfo.from_dict(v) 
            for k, v in data["metrics"].items()
        }


def _infer_dtype(value: Any) -> str:
    """Infer a simple type string from a value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


class MetricsDiscovery:
    """Discovers metrics from wandb runs."""
    
    # Keys to exclude from metrics (wandb internal keys)
    EXCLUDE_KEYS = {
        "_wandb", "_runtime", "_timestamp", "_step",
    }
    
    # Prefixes to exclude
    EXCLUDE_PREFIXES = ("system/", "system.", "_")
    
    # Media types to exclude (wandb stores these as dicts with _type field)
    EXCLUDE_MEDIA_TYPES = {
        "image-file", "images/separated", "video-file", "audio-file",
        "table", "table-file", "html", "html-file", "plotly", "plotly-file",
        "object3D-file", "molecule-file", "bokeh-file",
    }
    
    def __init__(
        self, 
        registry: Optional[MetricsRegistry] = None,
        include_system_metrics: bool = False,
    ):
        """
        Initialize discovery.
        
        Args:
            registry: Existing registry to update, or None to create new
            include_system_metrics: Whether to include system/wandb internal metrics
        """
        self.registry = registry or MetricsRegistry()
        self.include_system_metrics = include_system_metrics
    
    def _should_include(self, key: str) -> bool:
        """Check if a metric key should be included."""
        if self.include_system_metrics:
            return True
        
        if key in self.EXCLUDE_KEYS:
            return False
        
        for prefix in self.EXCLUDE_PREFIXES:
            if key.startswith(prefix):
                return False
        
        return True
    
    def discover_from_run(
        self, 
        run,
        scan_summary: bool = True,
        scan_history: bool = True,
        history_sample_size: int = 100,
    ):
        """
        Discover metrics from a single run.
        
        Args:
            run: Wandb Run object
            scan_summary: Whether to scan summary metrics
            scan_history: Whether to scan history (time-series) metrics
            history_sample_size: Number of history rows to sample
        """
        run_id = run.id
        
        # Skip if already scanned
        if run_id in self.registry._discovery_run_ids:
            return
        
        # Scan summary
        if scan_summary:
            # Handle both CachedRun (dict) and wandb Run (proxy object)
            if isinstance(run.summary, dict):
                summary = run.summary
            else:
                # For wandb Run objects, iterate keys safely
                summary = {}
                try:
                    for k in run.summary.keys():
                        try:
                            summary[k] = run.summary[k]
                        except Exception:
                            pass
                except Exception:
                    pass
            
            for key, value in summary.items():
                if not self._should_include(key):
                    continue
                
                # Skip media/artifact types (images, videos, tables, etc.)
                if isinstance(value, dict):
                    if "_type" in value:
                        media_type = value.get("_type", "")
                        if media_type in self.EXCLUDE_MEDIA_TYPES:
                            continue
                    # Skip any dict that looks like a wandb media object
                    if any(k in value for k in ("_type", "path", "artifact_path")):
                        continue
                
                self.registry.add_metric(
                    name=key,
                    source=MetricSource.SUMMARY,
                    dtype=_infer_dtype(value),
                    sample_value=value if not isinstance(value, (dict, list)) else None,
                )
        
        # Scan history (only works with wandb Run objects, not CachedRun)
        if scan_history and hasattr(run, 'history') and callable(run.history):
            try:
                history = run.history(samples=history_sample_size)
                if not history.empty:
                    for col in history.columns:
                        if not self._should_include(col):
                            continue
                        
                        # Get a sample value
                        sample = history[col].dropna()
                        sample_value = sample.iloc[0] if len(sample) > 0 else None
                        
                        # Skip complex types
                        if isinstance(sample_value, (dict, list)):
                            continue
                        
                        self.registry.add_metric(
                            name=col,
                            source=MetricSource.HISTORY,
                            dtype=_infer_dtype(sample_value),
                            sample_value=sample_value,
                            is_time_series=True,
                            typical_length=len(history),
                        )
            except Exception as e:
                # History might not be available for all runs
                pass
        
        self.registry._discovery_run_ids.add(run_id)
    
    def discover_from_runs(
        self,
        runs: list,
        scan_summary: bool = True,
        scan_history: bool = True,
        history_sample_size: int = 100,
        max_runs: Optional[int] = None,
        verbose: bool = False,
    ) -> MetricsRegistry:
        """
        Discover metrics from multiple runs.
        
        Args:
            runs: List of wandb Run objects
            scan_summary: Whether to scan summary metrics
            scan_history: Whether to scan history metrics
            history_sample_size: Number of history rows to sample per run
            max_runs: Maximum number of runs to scan (None = all)
            verbose: Print progress
            
        Returns:
            Updated MetricsRegistry
        """
        runs_to_scan = runs[:max_runs] if max_runs else runs
        
        for i, run in enumerate(runs_to_scan):
            if verbose:
                print(f"Scanning run {i+1}/{len(runs_to_scan)}: {run.name}")
            
            self.discover_from_run(
                run,
                scan_summary=scan_summary,
                scan_history=scan_history,
                history_sample_size=history_sample_size,
            )
        
        self.registry.last_discovery = datetime.now()
        return self.registry