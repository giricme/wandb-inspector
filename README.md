# wbutils

Inspect wandb projects: discover metrics, deduplicate runs, and generate reports with LaTeX tables.

## Features

- **Metrics Discovery**: Automatically discover available metrics from your wandb runs (both summary and time-series)
- **Run Deduplication**: Identify and filter duplicate runs based on configurable keys (task name, seed, learning rate, etc.)
- **Flexible Selection**: Choose among duplicates by latest run, longest run, or best metric value
- **Report Generation**: Generate structured reports with aggregated statistics
- **LaTeX Tables**: Export publication-ready LaTeX tables with proper formatting
- **Caching**: Cache discovered metrics for faster subsequent access

## Installation

```bash
# Install from source
pip install -e /path/to/wandb-inspector

# Or install dependencies manually
pip install wandb pandas
```

## Quick Start

### As a Python Library

```python
from wbutils.inspector import WandbInspector
from wbutils.dedup import DedupStrategy

# Initialize inspector
inspector = WandbInspector(entity="my-entity", project="my-project")

# Discover available metrics
inspector.discover_metrics(max_runs=50, verbose=True)
inspector.print_metrics()

# Get all runs
runs = inspector.get_runs()
print(f"Found {len(runs)} runs")

# Generate aggregated report
report = inspector.aggregate(
    metric_keys=["accuracy", "loss", "f1_score"],
    group_by="task_name"
)

# Export as LaTeX table
latex = inspector.to_latex_aggregated(
    report,
    caption="Results by Task",
    label="tab:results"
)
print(latex)

# Export to pandas DataFrame
df = inspector.to_dataframe(
    metric_keys=["accuracy", "loss"],
    config_keys=["task_name", "seed", "lr"]
)
df.to_csv("results.csv")
```

### Deduplication Strategies

```python
from wbutils.inspector import WandbInspector
from wbutils.dedup import DedupStrategy

# Enable deduplication by specifying keys
inspector = WandbInspector(
    "entity", "project",
    dedup_keys=["task_name", "seed"],  # Identify duplicates by these config keys
    dedup_strategy=DedupStrategy.LATEST  # Keep latest among duplicates
)

# Other strategies:
# DedupStrategy.LONGEST - keep run with most logged steps
# DedupStrategy.BEST - keep run with best metric value

# For BEST strategy, configure which metric to use:
inspector.set_dedup_strategy(
    DedupStrategy.BEST,
    metric_key="accuracy",
    higher_is_better=True
)

# Update dedup keys at runtime
inspector.set_dedup_keys(["task_name", "seed", "lr", "batch_size"])
```

### LaTeX Table Types

```python
# 1. Aggregated table (single group)
report = inspector.aggregate(["accuracy", "loss"])
latex = inspector.to_latex_aggregated(report)

# 2. Grouped table (by config key)
report = inspector.aggregate(["accuracy", "loss"], group_by="task_name")
latex = inspector.to_latex_aggregated(report)

# 3. Individual runs table
latex = inspector.to_latex_runs(
    metric_keys=["accuracy", "loss"],
    config_keys=["task_name", "seed"]
)

# 4. Comparison across experiments
inspector1 = WandbInspector("entity", "baseline-project")
inspector2 = WandbInspector("entity", "improved-project")

latex = WandbInspector.compare(
    {"Baseline": inspector1, "Ours": inspector2},
    metric_keys=["accuracy", "f1_score"],
    highlight_best=True,
    higher_is_better={"accuracy": True, "f1_score": True}
)
```

### Metrics Discovery

```python
from wbutils.discovery import MetricSource

# Discover metrics (scans run summaries and histories)
inspector.discover_metrics(
    max_runs=100,
    scan_summary=True,
    scan_history=True,
    refresh=False,  # Set True to re-scan
    verbose=True
)

# Filter metrics
summary_metrics = inspector.get_metrics(source=MetricSource.SUMMARY)
history_metrics = inspector.get_metrics(source=MetricSource.HISTORY)
common_metrics = inspector.get_metrics(min_run_count=10)
loss_metrics = inspector.get_metrics(pattern="loss")

# Just get metric names
metric_names = inspector.list_metric_names(min_run_count=5)
```

## Command Line Interface

```bash
# Discover metrics in a single project
wbutils discover my-entity my-project --max-runs 100 --verbose

# Discover metrics across multiple projects
wbutils discover my-entity project1 project2 project3 --max-runs 50

# List discovered metrics
wbutils metrics my-entity my-project --min-runs 5 --source summary

# List config keys (useful for finding dedup/group-by keys)
wbutils configs my-entity my-project --show-values

# Inspect a specific metric (see data points per run)
wbutils inspect my-entity my-project --metric eval/success
wbutils inspect my-entity my-project --metric eval/success --show-runs --max-runs 20

# Fetch and cache metric history (time series) - like discover but for history
wbutils history my-entity my-project --metric eval/success
wbutils history my-entity my-project --metric eval/success --refresh  # force re-fetch

# Export cached history to CSV
wbutils history my-entity my-project --metric eval/success -o eval_success.csv
wbutils history my-entity my-project --metric eval/success --config-keys env_name seed -o history.csv

# Plot cached history
wbutils plot my-entity my-project --metric eval/success                     # display
wbutils plot my-entity my-project --metric eval/success -o success.png      # save to file
wbutils plot my-entity my-project --metric eval/success --group-by env_name # group by env
wbutils plot my-entity my-project --metric eval/success --group-by env_name seed  # individual lines
wbutils plot my-entity my-project --metrics eval/success eval/return --group-by env_name  # subplots

# Split into multiple plots (one per env)
wbutils plot my-entity my-project --metric eval/success --split-by env_name -o curves.png

# Multi-project comparison
wbutils plot my-entity --metric eval/success \
  --project baseline:proj1 --project improved:proj2

# Multi-project with different metric names
wbutils plot my-entity --metric eval/success \
  --project DSRL:DSRL --project EXPO:EXPO \
  --metric-map EXPO=eval_base/success

# Customize plot
wbutils plot my-entity my-project --metric eval/success --group-by env_name \
  --title "Success Rate" --xlabel "Steps" --ylabel "Success" \
  --xlim 0 1000000 --ylim 0 1 --figsize 12 8

# Filter runs by config values
wbutils plot my-entity my-project --metric eval/success --filter env_name=square
wbutils history my-entity my-project --metric eval/success --filter env_name=square
wbutils report my-entity my-project --metrics eval/success --filter env_name=square

# Multiple filters (AND logic)
wbutils plot my-entity my-project --metric eval/success \
  --filter env_name=square --filter seed=0

# Filter with multiple values (OR within key)
wbutils plot my-entity my-project --metric eval/success \
  --filter env_name=square,tool_hang --filter seed=0,1,2

# Generate text report (default format)
wbutils report my-entity my-project --metrics eval/success

# Group by single key
wbutils report my-entity my-project --metrics eval/success --group-by env_name

# Group by multiple keys
wbutils report my-entity my-project --metrics eval/success --group-by env_name seed

# Generate LaTeX report
wbutils report my-entity my-project \
    --metrics eval/success eval/return \
    --group-by env_name \
    --format latex \
    --output results.tex

# Export to CSV
wbutils export my-entity my-project \
    --metrics eval/success \
    --config-keys env_name seed lr \
    --output results.csv

# Show project statistics
wbutils stats my-entity my-project

# Use deduplication
wbutils report my-entity my-project \
    --metrics eval/success \
    --dedup-keys env_name seed \
    --group-by env_name
```

## Filtering Pipeline

Commands process runs through a clear pipeline:

```
All runs
  → --filter (select runs matching config values)
    → --dedup-keys (deduplicate by config keys)
      → --group-by (aggregate for reports/plots)
        → --split-by (split into multiple plots)
```

**Filter syntax:**
```bash
--filter key=value              # exact match
--filter key=val1,val2          # OR: match any value
--filter key1=a --filter key2=b # AND: both must match
```

**Example workflow:**
```bash
# Filter to specific envs, dedup by seed, group by env
wbutils report my-entity my-project --metrics eval/success \
  --filter env_name=square,tool_hang \
  --dedup-keys env_name seed \
  --group-by env_name
```

## Example LaTeX Output

```latex
\begin{table}[htbp]
\centering
\caption{Results by Task}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
task\_name & accuracy & loss & f1\_score \\
\midrule
classification & $0.9234 \pm 0.0123$ & $0.2341 \pm 0.0456$ & $0.9156 \pm 0.0134$ \\
regression & $0.8765 \pm 0.0234$ & $0.3456 \pm 0.0567$ & $0.8654 \pm 0.0245$ \\
\bottomrule
\end{tabular}
\end{table}
```

## Caching

The inspector automatically caches **both runs and metrics** to `.cache/wbutils/` in the current working directory. This means most operations don't contact wandb unless you explicitly refresh:

```python
# Default: caches to .cache/wbutils/ in cwd
inspector = WandbInspector("entity", "project")

# These operations use cached data (no wandb API calls)
inspector.print_metrics()                    # Uses cached metrics registry
inspector.get_runs()                         # Uses cached run data
report = inspector.aggregate(["accuracy"])   # Uses cached runs
df = inspector.to_dataframe()                # Uses cached runs

# Force refresh from wandb API
inspector.discover_metrics(refresh=True)     # Re-fetch runs & re-scan metrics
inspector.get_runs(refresh=True)             # Re-fetch runs
report = inspector.aggregate(["accuracy"], refresh=True)

# Custom cache directory
inspector = WandbInspector("entity", "project", cache_dir="/path/to/cache")

# Disable caching entirely
inspector = WandbInspector("entity", "project", no_cache=True)
```

**What's cached:**
- Run metadata (id, name, config, summary, state, tags)
- Discovered metrics registry
- History data (per run + metric, via `wbutils history` command)

**What's NOT cached:**
- Images, videos, audio, tables, and other media artifacts

## Plotting Training Curves

### Quick Start

```bash
# Step 1: Fetch and cache history
wbutils history my-entity my-project --metric eval/success

# Step 2: Plot (uses cached data)
wbutils plot my-entity my-project --metric eval/success --group-by env_name
```

### Plot Examples

```bash
# Basic plot - mean ± std across all runs
wbutils plot my-entity my-project --metric eval/success

# Group by environment - one line per env with std bands
wbutils plot my-entity my-project --metric eval/success --group-by env_name

# Group by env + seed - individual lines (no std bands)
wbutils plot my-entity my-project --metric eval/success --group-by env_name seed

# Multiple metrics as subplots
wbutils plot my-entity my-project --metrics eval/success eval/return --group-by env_name

# Save to file instead of displaying
wbutils plot my-entity my-project --metric eval/success -o training_curves.png

# Full customization
wbutils plot my-entity my-project --metric eval/success --group-by env_name \
  --title "Success Rate by Environment" \
  --xlabel "Training Steps" \
  --ylabel "Success Rate" \
  --xlim 0 1000000 \
  --ylim 0 1 \
  --figsize 12 8 \
  -o success.png
```

### Split into Multiple Plots

Use `--split-by` to create separate plots for each value of a config key:

```bash
# One plot per environment (saves: success_square.png, success_tool_hang.png, etc.)
wbutils plot my-entity my-project --metric eval/success --split-by env_name -o success.png

# Split by env, group by seed within each plot
wbutils plot my-entity my-project --metric eval/success --split-by env_name --group-by seed -o curves.png
```

### Multi-Project Comparison

Compare results across different projects in the same plot:

```bash
# Cache history for each project first
wbutils history my-entity baseline-proj --metric eval/success
wbutils history my-entity improved-proj --metric eval/success

# Compare with labeled projects
wbutils plot my-entity --metric eval/success \
  --project baseline:baseline-proj \
  --project improved:improved-proj

# Multi-project + group by env
wbutils plot my-entity --metric eval/success \
  --project baseline:baseline-proj \
  --project improved:improved-proj \
  --group-by env_name

# Multi-project + split by env (one plot per env comparing methods)
wbutils plot my-entity --metric eval/success \
  --project baseline:baseline-proj \
  --project improved:improved-proj \
  --split-by env_name \
  -o comparison.png
```

### Metric Name Mapping

When comparing projects that use different metric names for the same thing, use `--metric-map`:

```bash
# EXPO uses eval_base/success instead of eval/success
wbutils plot my-entity --metric eval/success \
  --project DSRL:DSRL \
  --project EXPO:EXPO \
  --metric-map EXPO=eval_base/success

# For multiple metrics, specify each mapping
wbutils plot my-entity --metrics eval/success eval/return \
  --project DSRL:DSRL \
  --project EXPO:EXPO \
  --metric-map EXPO:eval/success=eval_base/success \
  --metric-map EXPO:eval/return=eval_base/return
```

Note: You need to cache history for the actual metric names:
```bash
wbutils history my-entity DSRL --metric eval/success
wbutils history my-entity EXPO --metric eval_base/success  # actual name in EXPO
```

### Manual Plotting with CSV Export

If you need more control, export to CSV and use matplotlib/seaborn directly:

```bash
wbutils history my-entity my-project --metric eval/success --config-keys env_name seed -o history.csv
```

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv("history.csv")
sns.lineplot(data=df, x="step", y="eval/success", hue="env_name")
```

### Python API

```python
from wbutils.inspector import WandbInspector

inspector = WandbInspector("my-entity", "my-project")

# Get cached history (after running `wbutils history`)
history = inspector.get_history("eval/success")
# [{"run_id": "abc", "run_name": "run1", "step": 1000, "value": 0.5}, ...]

# Check what's cached
metrics = inspector.list_cached_history_metrics()
# ["eval/success", "eval/return"]
```

## API Reference

### WandbInspector

Main class for inspecting wandb projects.

| Method | Description |
|--------|-------------|
| `discover_metrics()` | Discover available metrics from runs |
| `get_metrics()` | Get discovered metrics with filters |
| `print_metrics()` | Print formatted metrics table |
| `fetch_runs()` | Fetch runs from wandb |
| `get_runs()` | Get runs (optionally deduplicated) |
| `get_duplicate_stats()` | Get deduplication statistics |
| `aggregate()` | Generate aggregated statistics |
| `run_reports()` | Generate individual run reports |
| `to_latex_aggregated()` | Generate LaTeX from aggregated report |
| `to_latex_runs()` | Generate LaTeX table of runs |
| `to_dataframe()` | Export to pandas DataFrame |
| `get_history()` | Get cached history data for a metric |
| `has_history()` | Check if history is cached for a metric |
| `list_cached_history_metrics()` | List metrics with cached history |
| `compare()` | Compare multiple inspectors (class method) |

### DedupConfig

| Field | Type | Description |
|-------|------|-------------|
| `keys` | `list[str]` | Config keys for identifying duplicates |
| `strategy` | `DedupStrategy` | Selection strategy |
| `metric_key` | `str` | Metric for BEST strategy |
| `metric_higher_is_better` | `bool` | Direction for BEST strategy |
| `include_incomplete` | `bool` | Include runs missing dedup keys |

### DedupStrategy

| Value | Description |
|-------|-------------|
| `LATEST` | Select most recently created run |
| `LONGEST` | Select run with most logged steps |
| `BEST` | Select run with best metric value |
| `CUSTOM` | Use custom selector function |

## License

MIT