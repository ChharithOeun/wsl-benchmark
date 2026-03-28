"""Tests for wsl_benchmark -- CPU-only, no GPU required."""
import json
import subprocess
import sys


def test_import():
    import wsl_benchmark
    assert wsl_benchmark.__version__ == "0.1.0"
    assert callable(wsl_benchmark.run_benchmark)
    assert callable(wsl_benchmark.get_results)


def test_run_benchmark_cpu():
    import wsl_benchmark
    data = wsl_benchmark.run_benchmark(ops=["matmul"], size=64, warmup=1, runs=2)
    assert "results" in data
    assert "device" in data
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["op"] == "matmul"
    assert r["error"] is None
    assert r["median_ms"] is not None and r["median_ms"] > 0


def test_run_benchmark_all_ops():
    import wsl_benchmark
    data = wsl_benchmark.run_benchmark(ops=["matmul", "fft"], size=64, warmup=1, runs=1)
    assert len(data["results"]) == 2
    for r in data["results"]:
        assert "op" in r
        assert "median_ms" in r


def test_json_output_format():
    result = subprocess.run(
        [sys.executable, "-m", "wsl_benchmark",
         "--ops", "matmul", "--size", "64", "--warmup", "1", "--runs", "1", "--json"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Exit {result.returncode}: {result.stderr}"
    data = json.loads(result.stdout)
    assert "results" in data
    assert "device" in data
    r = data["results"][0]
    assert r["op"] == "matmul"
    assert r["median_ms"] is not None


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "wsl_benchmark", "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "benchmark" in result.stdout.lower()


def test_cli_invalid_op():
    result = subprocess.run(
        [sys.executable, "-m", "wsl_benchmark", "--ops", "invalid_op"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "Unknown ops" in result.stderr


def test_cli_invalid_size():
    result = subprocess.run(
        [sys.executable, "-m", "wsl_benchmark", "--ops", "matmul", "--size", "9999"],
        capture_output=True, text=True,
    )
    assert result.returncode == 1


def test_report_format():
    from wsl_benchmark.report import format_table, format_json
    import wsl_benchmark
    data = wsl_benchmark.run_benchmark(ops=["matmul"], size=64, warmup=1, runs=1)
    table = format_table(data)
    assert "matmul" in table
    assert "WSL-BENCHMARK" in table
    js = format_json(data)
    parsed = json.loads(js)
    assert "results" in parsed
