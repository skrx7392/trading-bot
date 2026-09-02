# tbot

Systematic-trading research pipeline.

## Layout

- `src/tbot/` — library code
- `tests/` — pytest suite
- `data/` — all generated data (gitignored); override the location with `TBOT_DATA`
- `docs/` — design spec and implementation plans

## Development

```bash
uv sync
uv run pytest            # unit tests (integration tests deselected by default)
uv run pytest -m integration   # tests needing network/local services
```
