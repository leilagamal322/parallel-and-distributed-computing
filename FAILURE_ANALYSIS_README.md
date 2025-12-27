# Failure Performance Analysis

This tool collects and presents performance metrics before, during, and after failures in the distributed particle simulation system.

## Features

- **Throughput (req/sec)**: Shows dip during failure, recovery afterward
- **Latency (p95)**: Shows spike → return to baseline
- **Recovery time**: Seconds until normal behavior
- **Annotated graphs**: Latency vs Time and Throughput vs Time with failure/recovery markers

## Usage

### Basic Usage

```bash
python run_failure_analysis.py --replicas 3 --rate 10.0 --duration 120
```

### With Custom Failure Scenarios

```bash
python run_failure_analysis.py --replicas 3 --rate 10.0 --duration 120 --scenarios example_failure_scenarios.json
```

### Command Line Options

- `--replicas, -n`: Number of replicas (default: 3)
- `--rate, -r`: Requests per second (default: 10.0)
- `--duration, -d`: Duration in seconds (default: 120.0)
- `--output-dir, -o`: Output directory for plots (default: plots)
- `--scenarios, -s`: JSON file with failure scenarios (optional)

## Failure Scenario Format

Create a JSON file with failure scenarios:

```json
[
  {
    "type": "kill",
    "replica_id": 0,
    "time": 30.0,
    "recover": true,
    "recover_after": 10.0,
    "description": "Kill replica 0 at 30s, recover after 10s"
  }
]
```

Fields:
- `type`: Type of failure (currently only "kill" supported)
- `replica_id`: ID of replica to fail (0-based)
- `time`: Time in seconds from start to inject failure
- `recover`: Whether to automatically recover the replica
- `recover_after`: Seconds to wait before recovering
- `description`: Optional description

## Output

The tool generates:

1. **latency_vs_time.png**: P95 latency over time with failure/recovery annotations
2. **throughput_vs_time.png**: Throughput over time with failure/recovery annotations
3. **performance_with_failures.png**: Combined plot with both metrics

All plots are saved to the `plots/` directory (or specified output directory).

## Metrics Collected

### Before Failures
- Baseline P95 latency
- Baseline throughput

### During Failures
- Latency spikes
- Throughput dips
- Failure timestamps

### After Failures
- Recovery timestamps
- Time to return to baseline metrics
- Overall recovery time

## Example Output

```
Performance Analysis Summary
================================================================================

Baseline Metrics (before failures):
  P95 Latency: 45.23 ms
  Throughput: 9.87 req/sec

Recovery Times:
   Failure Time | Recovery Time (s) | Metric Recovery (s) |    Replica
--------------------------------------------------------------------------------
           30.0 |             10.00 |              12.50 |          0
           60.0 |              8.00 |               9.75 |          1

Average Recovery Time: 9.00 seconds
Average Metric Recovery Time: 11.13 seconds
```

## Requirements

- Python 3.7+
- matplotlib
- numpy
- All dependencies from requirements.txt

## Notes

- The system needs time to stabilize before failures are injected (5 seconds)
- Recovery time calculation uses a 20% tolerance for baseline comparison
- Metrics are collected in 1-second windows for time series analysis

