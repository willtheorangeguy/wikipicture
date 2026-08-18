# WikiPicture — Development

## Setup

```bash
pip install -e ".[dev]"
```

## Commands

```bash
python -m pytest tests/ -v --tb=short
python -m pytest tests/test_scorer.py -v
python -m pytest tests/test_scorer.py::TestScoredArticle::test_score_breakdown -v
python -m pytest tests/ --cov=wikipicture --cov-report=html

python main.py scan /path/to/photos --output report.html
python main.py clear-cache --older-than 30
```

`main.py` prepends `src/` to `sys.path`, so the CLI runs without installing.

## Tests

Tests mirror module names: `test_scorer.py`, `test_geocoder.py`, `test_wiki_analyzer.py`,
`test_commons_checker.py`, `test_exif_extractor.py`, `test_quality_filter.py`,
`test_network_paths.py`.

**Every HTTP call is mocked with `responses`.** No test touches the network, so the suite is fast
and deterministic, and running it does not consume anyone's rate limit.

`cli.py` itself is uncovered — it is orchestration. The decisions worth testing live in the stage
modules, which is why they are separate.

## Conventions

- **One module per pipeline stage**, each with its own dataclass as output. A new stage should
  follow the pattern rather than growing `cli.py`.
- **All HTTP through `http_client.make_session()`.** Never construct a bare session: it would
  miss the User-Agent and the retry policy.
- **Rate limits are not optional.** `geocoder.py` enforces one request per second. Do not add a
  path around it.
- **Platform specifics stay isolated.** Windows UNC handling lives in `network_paths.py`, guarded
  by `sys.platform == 'win32'`.
- **Optional imports degrade.** `tqdm` is wrapped in `try/except ImportError` so a missing
  progress bar is not a crash; keep that shape for anything else optional.

## Adding a scoring signal

`scorer.py` components each return `(score, reason)`. Keep that: the report's value is the
explanation, not the number. If you add a component, adjust the others so the total stays 100 —
the recommendation thresholds (70/45/25) assume it.

## CI

`.github/workflows/ci.yml` runs the suite on Python 3.10, 3.11, and 3.12.
`.github/workflows/publish.yml` publishes to PyPI with OIDC trusted publishing.

## Recording defects

Bugs found while working here go in [`internal/known-issues.md`](./internal/known-issues.md)
rather than being fixed in passing, unless fixing them is the job you are on.
