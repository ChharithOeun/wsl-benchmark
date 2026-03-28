# Use Cases

## Verify GPU is faster than CPU

```bash
pip install torch wsl-benchmark
python -m wsl_benchmark --json | python3 -c "
import json, sys
d = json.load(sys.stdin)
print('Device:', d['device'])
for r in d['results']:
    print(r['op'], r['median_ms'], 'ms')
"
```

## Automated CI benchmark (saves results to file)

```bash
python -m wsl_benchmark --json > benchmark_results.json
```

## Compare before/after a PyTorch version upgrade

```bash
# Before upgrade:
python -m wsl_benchmark --json > before.json

# After upgrade:
pip install --upgrade torch
python -m wsl_benchmark --json > after.json

python3 -c "
import json
before = json.load(open('before.json'))['results']
after = json.load(open('after.json'))['results']
for b, a in zip(before, after):
    diff = ((a['median_ms'] - b['median_ms']) / b['median_ms']) * 100
    print(f\"{b['op']}: {b['median_ms']:.1f}ms -> {a['median_ms']:.1f}ms ({diff:+.1f}%)\")
"
```

## Stress test (many runs)

```bash
python -m wsl_benchmark --warmup 5 --runs 50 --size 2048
```

## Quick sanity check (fast)

```bash
python -m wsl_benchmark --ops matmul --size 128 --warmup 1 --runs 3
```
