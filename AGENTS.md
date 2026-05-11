# Project Guidelines

## Source Of Truth

- This file is the current project-wide guide for coding agents working in this repository.
- Trust current code, tests, and [README.md](README.md) over older design notes or stale comments.
- [agent_instructions.txt](agent_instructions.txt) is deprecated and exists only as a redirect.

## Architecture

- Primary entry point: `run.py`.
- Live production flow: `prompt -> classify/bootstrap -> conductor -> editor -> JSON -> translate -> TTS -> audio`, with optional publish, video, package, analysis, emotion-flow, audition, series, and university modes routed by the CLI.
- Shipped show types: debates, news, podcast, audiobooks, lecture.
- Live package layout:
  - `vox_romana/core/`: shared LLM client, config, session, routing, editing, duration, and conductor infrastructure.
  - `vox_romana/shows/`: per-show bootstrap + conductor pipelines.
  - `vox_romana/tts/`, `vox_romana/translate/`, `vox_romana/publish/`, `vox_romana/animation/`, `vox_romana/memory/`, `vox_romana/university/`, `vox_romana/analysis/`, `vox_romana/emotion_flow/`, `vox_romana/cli/`: downstream render, analysis, and orchestration layers.
- Inter-agent runtime messaging lives in `vox_romana/core/messages.py` via `AgentMessage`, `MessageType`, and `AgentRouter`.
- `BaseConductor` in `vox_romana/core/conductor.py` is the shared runtime abstraction today, including some scripted flows. If a docstring mentions a future `BaseRenderer`, treat that as planned work, not a shipped class.
- The older two-step CLI (`run_show.py` / `render_show.py`) is not part of the current repo. Use `run.py`.
- MCP entry point: `python -m vox_romana.mcp`.

## Build And Test

- Python: 3.12+.
- Environment setup:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Preflight check: `./.venv/bin/python preflight.py`
- Unit tests: `./.venv/bin/python -m pytest -q`
- Narrow test example: `./.venv/bin/python -m pytest tests/test_audiobook_conductor.py -q`
- Integration tests are opt-in: `./.venv/bin/python -m pytest --run-integration -q`
- `conftest.py` promotes warnings to errors under pytest. Treat new warnings as failures.

## Conventions

- Keep changes schema-preserving unless the task explicitly changes output contracts. Generated JSON feeds translation, TTS, publish, package, and animation code.
- When touching conductors, bootstraps, routing, or shared core infrastructure, validate with the narrowest relevant tests first, then widen only if needed.
- Prefer current repo docs over copying architecture into new files. For deeper operational reference, see [README.md](README.md), [docs/local_lens_rack.md](docs/local_lens_rack.md), and [docs/local_stack.md](docs/local_stack.md).
- Use `vox_romana.yaml` and `vox_romana.example.yaml` for project defaults; avoid hard-coding machine-local paths unless the surrounding code already depends on them.