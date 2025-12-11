"""Report generation with structured output and LaTeX tables."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class MetricStats:
    """Statistics for a metric across runs."""

    name: str
    count: int = 0
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    median: Optional[float] = None
    values: list = field(default_factory=list)


@dataclass
class RunReport:
    """Report for a single run."""

    run_id: str
    run_name: str
    config: dict
    metrics: dict[str, Any]
    created_at: str
    state: str
    step_count: Optional[int] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class AggregatedReport:
    """Aggregated report across multiple runs."""

    total_runs: int
    metric_stats: dict[str, MetricStats]
    config_values: dict[str, list]  # Unique values for each config key
    group_by_keys: Optional[list[str]] = None  # Changed from group_by_key
    groups: Optional[dict] = None  # Grouped stats if group_by_keys is set


def compute_stats(values: list) -> MetricStats:
    """Compute statistics for a list of numeric values."""
    import statistics

    # Filter to numeric values only
    numeric = [v for v in values if isinstance(v, (int, float)) and v is not None]

    stats = MetricStats(name="", values=numeric, count=len(numeric))

    if len(numeric) > 0:
        stats.mean = statistics.mean(numeric)
        stats.min = min(numeric)
        stats.max = max(numeric)
        stats.median = statistics.median(numeric)

        if len(numeric) > 1:
            stats.std = statistics.stdev(numeric)

    return stats


class ReportGenerator:
    """Generates reports from wandb runs."""

    def __init__(self, runs: list):
        """
        Initialize with a list of runs.

        Args:
            runs: List of wandb Run objects
        """
        self.runs = runs

    def generate_run_reports(
        self,
        metric_keys: Optional[list[str]] = None,
        config_keys: Optional[list[str]] = None,
    ) -> list[RunReport]:
        """
        Generate individual run reports.

        Args:
            metric_keys: Specific metrics to include (None = all)
            config_keys: Specific config keys to include (None = all)

        Returns:
            List of RunReport objects
        """
        reports = []

        for run in self.runs:
            summary = run.summary if isinstance(run.summary, dict) else dict(run.summary)
            config = run.config if isinstance(run.config, dict) else dict(run.config)

            # Filter metrics
            if metric_keys:
                metrics = {k: summary.get(k) for k in metric_keys if k in summary}
            else:
                metrics = {
                    k: v
                    for k, v in summary.items()
                    if not k.startswith("_") and not isinstance(v, dict)
                }

            # Filter config
            if config_keys:
                filtered_config = {k: config.get(k) for k in config_keys}
            else:
                filtered_config = config

            # Get step count
            step_count = summary.get("_step")

            reports.append(
                RunReport(
                    run_id=run.id,
                    run_name=run.name,
                    config=filtered_config,
                    metrics=metrics,
                    created_at=run.created_at,
                    state=run.state,
                    step_count=step_count,
                    tags=run.tags,
                )
            )

        return reports

    def generate_aggregated_report(
        self,
        metric_keys: list[str],
        config_keys: Optional[list[str]] = None,
        group_by: Optional[list[str] | str] = None,
    ) -> AggregatedReport:
        """
        Generate aggregated statistics across runs.

        Args:
            metric_keys: Metrics to aggregate
            config_keys: Config keys to track unique values
            group_by: Config key(s) to group results by (str or list of str)

        Returns:
            AggregatedReport object
        """
        # Normalize group_by to list
        if isinstance(group_by, str):
            group_by_keys = [group_by]
        elif group_by:
            group_by_keys = list(group_by)
        else:
            group_by_keys = None

        # Collect metric values
        metric_values = defaultdict(list)
        config_values = defaultdict(set)
        grouped_metrics = defaultdict(lambda: defaultdict(list))

        for run in self.runs:
            summary = run.summary if isinstance(run.summary, dict) else dict(run.summary)
            config = run.config if isinstance(run.config, dict) else dict(run.config)

            # Get group key if grouping (tuple for multiple keys)
            group_val = None
            if group_by_keys:
                group_val = tuple(config.get(k, "unknown") for k in group_by_keys)

            # Collect metrics
            for key in metric_keys:
                if key in summary:
                    val = summary[key]
                    metric_values[key].append(val)

                    if group_by_keys:
                        grouped_metrics[group_val][key].append(val)

            # Collect config values
            keys_to_check = config_keys or list(config.keys())
            for key in keys_to_check:
                if key in config:
                    val = config[key]
                    if isinstance(val, (str, int, float, bool)):
                        config_values[key].add(val)

        # Compute stats
        metric_stats = {}
        for key, values in metric_values.items():
            stats = compute_stats(values)
            stats.name = key
            metric_stats[key] = stats

        # Compute grouped stats if grouping
        groups = None
        if group_by_keys:
            groups = {}
            for group_val, group_metrics in grouped_metrics.items():
                groups[group_val] = {}
                for key, values in group_metrics.items():
                    stats = compute_stats(values)
                    stats.name = key
                    groups[group_val][key] = stats

        return AggregatedReport(
            total_runs=len(self.runs),
            metric_stats=metric_stats,
            config_values={k: sorted(list(v)) for k, v in config_values.items()},
            group_by_keys=group_by_keys,
            groups=groups,
        )


class LaTeXFormatter:
    """Formats reports as LaTeX tables."""

    @staticmethod
    def escape_latex(text: str) -> str:
        """Escape special LaTeX characters."""
        if not isinstance(text, str):
            text = str(text)

        replacements = {
            "_": r"\_",
            "%": r"\%",
            "&": r"\&",
            "#": r"\#",
            "$": r"\$",
            "{": r"\{",
            "}": r"\}",
            "^": r"\^{}",
            "~": r"\~{}",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def format_number(val: Any, precision: int = 4) -> str:
        """Format a number for display."""
        if val is None:
            return "-"
        if isinstance(val, float):
            if abs(val) < 0.0001 or abs(val) > 10000:
                return f"{val:.{precision}e}"
            return f"{val:.{precision}f}"
        return str(val)

    def runs_table(
        self,
        reports: list[RunReport],
        metric_keys: list[str],
        config_keys: Optional[list[str]] = None,
        caption: str = "Run Results",
        label: str = "tab:runs",
        precision: int = 4,
    ) -> str:
        """
        Generate a LaTeX table of individual runs.

        Args:
            reports: List of RunReport objects
            metric_keys: Metrics to include as columns
            config_keys: Config keys to include as columns
            caption: Table caption
            label: LaTeX label
            precision: Decimal precision for numbers

        Returns:
            LaTeX table string
        """
        config_keys = config_keys or []

        # Build column spec
        num_cols = 1 + len(config_keys) + len(metric_keys)  # Run name + configs + metrics
        col_spec = "l" + "c" * (num_cols - 1)

        # Build header
        headers = ["Run"]
        headers.extend([self.escape_latex(k) for k in config_keys])
        headers.extend([self.escape_latex(k) for k in metric_keys])

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        # Build rows
        for report in reports:
            row = [self.escape_latex(report.run_name[:20])]  # Truncate name

            for key in config_keys:
                val = report.config.get(key, "-")
                row.append(self.format_number(val, precision))

            for key in metric_keys:
                val = report.metrics.get(key)
                row.append(self.format_number(val, precision))

            lines.append(" & ".join(row) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def aggregated_table(
        self,
        report: AggregatedReport,
        metric_keys: Optional[list[str]] = None,
        show_std: bool = True,
        show_minmax: bool = False,
        caption: str = "Aggregated Results",
        label: str = "tab:aggregated",
        precision: int = 4,
    ) -> str:
        """
        Generate a LaTeX table of aggregated statistics.

        Args:
            report: AggregatedReport object
            metric_keys: Metrics to include (None = all)
            show_std: Include standard deviation
            show_minmax: Include min/max columns
            caption: Table caption
            label: LaTeX label
            precision: Decimal precision

        Returns:
            LaTeX table string
        """
        keys = metric_keys or list(report.metric_stats.keys())

        # Build columns
        headers = ["Metric", "Mean"]
        if show_std:
            headers.append("Std")
        if show_minmax:
            headers.extend(["Min", "Max"])
        headers.append("N")

        col_spec = "l" + "c" * (len(headers) - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        for key in keys:
            if key not in report.metric_stats:
                continue

            stats = report.metric_stats[key]
            row = [self.escape_latex(key)]
            row.append(self.format_number(stats.mean, precision))

            if show_std:
                row.append(self.format_number(stats.std, precision))
            if show_minmax:
                row.append(self.format_number(stats.min, precision))
                row.append(self.format_number(stats.max, precision))

            row.append(str(stats.count))
            lines.append(" & ".join(row) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def grouped_table(
        self,
        report: AggregatedReport,
        metric_keys: Optional[list[str]] = None,
        show_std: bool = True,
        caption: str = "Results by Group",
        label: str = "tab:grouped",
        precision: int = 4,
        transpose: bool = False,
    ) -> str:
        """
        Generate a LaTeX table grouped by config key(s).

        Args:
            report: AggregatedReport with groups
            metric_keys: Metrics to include
            show_std: Show mean±std format
            caption: Table caption
            label: LaTeX label
            precision: Decimal precision
            transpose: If True, groups as rows, metrics as columns

        Returns:
            LaTeX table string
        """
        if not report.groups:
            return "% No groups to display"

        keys = metric_keys or list(report.metric_stats.keys())
        groups = sorted(report.groups.keys())
        group_by_keys = report.group_by_keys or ["Group"]

        # Format group value for display (handle tuples)
        def format_group(g):
            if isinstance(g, tuple):
                return ", ".join(str(v) for v in g)
            return str(g)

        if transpose:
            # Groups as rows, metrics as columns
            # For multiple group keys, create one header column per key
            headers = [self.escape_latex(k) for k in group_by_keys]
            headers.extend([self.escape_latex(k) for k in keys])

            col_spec = "l" * len(group_by_keys) + "c" * len(keys)

            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
            ]

            for group in groups:
                # Group key columns
                if isinstance(group, tuple):
                    row = [self.escape_latex(str(v)) for v in group]
                else:
                    row = [self.escape_latex(str(group))]

                # Metric columns
                for key in keys:
                    stats = report.groups[group].get(key)
                    if stats and stats.mean is not None:
                        if show_std and stats.std is not None:
                            val = f"${self.format_number(stats.mean, precision)} \\pm {self.format_number(stats.std, precision)}$"
                        else:
                            val = self.format_number(stats.mean, precision)
                    else:
                        val = "-"
                    row.append(val)
                lines.append(" & ".join(row) + r" \\")
        else:
            # Metrics as rows, groups as columns
            # Format group values for headers
            def format_group_header(g):
                if isinstance(g, tuple):
                    return ", ".join(str(v) for v in g)
                return str(g)

            headers = ["Metric"]
            headers.extend([self.escape_latex(format_group_header(g)) for g in groups])

            col_spec = "l" + "c" * len(groups)

            lines = [
                r"\begin{table}[htbp]",
                r"\centering",
                f"\\caption{{{caption}}}",
                f"\\label{{{label}}}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
                " & ".join(headers) + r" \\",
                r"\midrule",
            ]

            for key in keys:
                row = [self.escape_latex(key)]
                for group in groups:
                    stats = report.groups[group].get(key)
                    if stats and stats.mean is not None:
                        if show_std and stats.std is not None:
                            val = f"${self.format_number(stats.mean, precision)} \\pm {self.format_number(stats.std, precision)}$"
                        else:
                            val = self.format_number(stats.mean, precision)
                    else:
                        val = "-"
                    row.append(val)
                lines.append(" & ".join(row) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def comparison_table(
        self,
        reports: dict[str, AggregatedReport],
        metric_keys: list[str],
        show_std: bool = True,
        caption: str = "Method Comparison",
        label: str = "tab:comparison",
        precision: int = 4,
        highlight_best: bool = True,
        higher_is_better: Optional[dict[str, bool]] = None,
    ) -> str:
        """
        Generate a comparison table across multiple experiment sets.

        Args:
            reports: Dict mapping method/experiment name to AggregatedReport
            metric_keys: Metrics to compare
            show_std: Show mean±std format
            caption: Table caption
            label: LaTeX label
            precision: Decimal precision
            highlight_best: Bold the best value for each metric
            higher_is_better: Dict mapping metric name to whether higher is better

        Returns:
            LaTeX table string
        """
        higher_is_better = higher_is_better or {}
        methods = list(reports.keys())

        headers = ["Metric"]
        headers.extend([self.escape_latex(m) for m in methods])

        col_spec = "l" + "c" * len(methods)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(headers) + r" \\",
            r"\midrule",
        ]

        for key in metric_keys:
            row = [self.escape_latex(key)]

            # Collect values for highlighting
            values = []
            for method in methods:
                stats = reports[method].metric_stats.get(key)
                if stats and stats.mean is not None:
                    values.append((method, stats.mean, stats.std))
                else:
                    values.append((method, None, None))

            # Find best
            best_method = None
            if highlight_best:
                valid = [(m, v) for m, v, _ in values if v is not None]
                if valid:
                    hib = higher_is_better.get(key, True)
                    if hib:
                        best_method = max(valid, key=lambda x: x[1])[0]
                    else:
                        best_method = min(valid, key=lambda x: x[1])[0]

            # Format values
            for method, mean, std in values:
                if mean is not None:
                    if show_std and std is not None:
                        val = f"${self.format_number(mean, precision)} \\pm {self.format_number(std, precision)}$"
                    else:
                        val = self.format_number(mean, precision)

                    if highlight_best and method == best_method:
                        val = f"\\textbf{{{val}}}"
                else:
                    val = "-"
                row.append(val)

            lines.append(" & ".join(row) + r" \\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)
