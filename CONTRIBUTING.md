# Contributing to Context Bucket

Thank you for your interest in contributing. This project is local-first by design: no cloud vector DBs, no API keys required for core usage.

## Development setup

```bash
git clone https://github.com/crackdevbuild/context-bucket.git
cd context-bucket
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Running tests

```bash
pytest -q
```

The first run downloads the ONNX MiniLM embedding model (~500MB) into your Hugging Face cache.

## Running benchmarks locally

Benchmark outputs are gitignored. To reproduce results:

```bash
python benchmark/generate_datasets.py
python benchmark/generate_cases.py
python benchmark/run_benchmark.py --variant structured --series 10 --runs-per-series 15
python benchmark/generate_comparison_report.py
```

Published README evidence uses static images in `docs/images/` and aggregate summaries in `benchmark/run_summary_*.json`. Regenerate chart PNGs with:

```bash
pip install matplotlib
python benchmark/export_readme_charts.py
```

## Pull requests

1. Fork the repo and create a branch from `main`.
2. Keep changes focused; match existing code style.
3. Add or update tests for behavior changes.
4. Ensure `pytest -q` passes locally.
5. Open a PR with a clear description of what changed and why.

## Reporting issues

Use [GitHub Issues](https://github.com/crackdevbuild/context-bucket/issues) for bugs, benchmark discrepancies, and feature requests. Include Python version, OS, and minimal reproduction steps when possible.

## Publishing to PyPI (maintainers)

PyPI distribution is not automated yet. To publish a release manually:

```bash
pip install build twine
python -m build
twine upload dist/*
```

Requires a [PyPI API token](https://pypi.org/manage/account/token/) configured via `~/.pypirc` or `TWINE_USERNAME` / `TWINE_PASSWORD` environment variables.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
