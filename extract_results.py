#!/usr/bin/env python3
import re
import statistics

runs = {
    'baseline_seed42': 'logs/baseline_seed42/log_metrics.txt',
    'baseline_seed43': 'logs/baseline_seed43/log_metrics.txt',
    'baseline_seed44': 'logs/baseline_seed44/log_metrics.txt',
    'softfilter_seed42': 'logs/softfilter_seed42/log_metrics.txt',
    'softfilter_seed43': 'logs/softfilter_seed43/log_metrics.txt',
    'softfilter_seed44': 'logs/softfilter_seed44/log_metrics.txt',
}

results = {}
for name, path in runs.items():
    with open(path, 'r') as f:
        content = f.read()
    # Find the last test F1 value (handles tabs and spaces)
    matches = re.findall(r'test\nP: [^\n]+F1: ([\d.]+)%', content)
    if matches:
        results[name] = float(matches[-1])
        print(f"{name}: {matches[-1]}%")

# Compute statistics
baseline_scores = [results['baseline_seed42'], results['baseline_seed43'], results['baseline_seed44']]
softfilter_scores = [results['softfilter_seed42'], results['softfilter_seed43'], results['softfilter_seed44']]

print("\n=== BASELINE (3 seeds) ===")
print(f"  Score 42: {baseline_scores[0]:.2f}%")
print(f"  Score 43: {baseline_scores[1]:.2f}%")
print(f"  Score 44: {baseline_scores[2]:.2f}%")
print(f"  Mean: {statistics.mean(baseline_scores):.2f}%")
print(f"  Std Dev: {statistics.stdev(baseline_scores):.2f}%")

print("\n=== SOFT FILTER #3 (3 seeds) ===")
print(f"  Score 42: {softfilter_scores[0]:.2f}%")
print(f"  Score 43: {softfilter_scores[1]:.2f}%")
print(f"  Score 44: {softfilter_scores[2]:.2f}%")
print(f"  Mean: {statistics.mean(softfilter_scores):.2f}%")
print(f"  Std Dev: {statistics.stdev(softfilter_scores):.2f}%")

print("\n=== COMPARISON ===")
diff = statistics.mean(softfilter_scores) - statistics.mean(baseline_scores)
print(f"  Difference: {diff:+.2f}%")
print(f"  Baseline std: {statistics.stdev(baseline_scores):.2f}%")
print(f"  Soft filter std: {statistics.stdev(softfilter_scores):.2f}%")
