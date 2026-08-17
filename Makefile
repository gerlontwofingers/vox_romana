# vox_romana Makefile
# ──────────────────────────────────────────────────────────────────────────────
# Override the Python interpreter if needed:
#   make install PYTHON=python3.12
#   make test    PYTHON=/opt/homebrew/bin/python3
#
# The default uses the project .venv when it exists, falling back to python3.
# ──────────────────────────────────────────────────────────────────────────────

PYTHON ?= $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.DEFAULT_GOAL := help

.PHONY: help install check test test-integration models clean

# ── help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "vox_romana — available targets"
	@echo "──────────────────────────────────────────────────────"
	@echo "  make install   Create .venv and install requirements.txt"
	@echo "  make check     Run preflight.py (environment health check)"
	@echo "  make test      Run the full pytest suite"
	@echo "  make test-integration  Run live LLM integration tests (requires DEEPSEEK_API_KEY)"
	@echo "  make models    Download Piper voice models via git-lfs"
	@echo "  make clean     Remove .venv, __pycache__, and pytest artefacts"
	@echo ""
	@echo "Override Python:  make install PYTHON=python3.12"
	@echo ""

# ── install ───────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Done. Activate with:  source .venv/bin/activate"
	@echo "Then run:  make check"

# ── check ─────────────────────────────────────────────────────────────────────
check:
	$(PYTHON) preflight.py

# ── test ──────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest -n auto --timeout=30

# Stream live output to logs/test_run.log while also printing to stdout.
# Usage:  make test-log          (parallel, default)
#         make test-log N=1      (serial, for debugging a specific failure)
N ?= auto
test-log:
	@mkdir -p logs
	$(PYTHON) -m pytest -n $(N) --timeout=30 --tb=short 2>&1 | tee logs/test_run.log

# ── test-integration ─────────────────────────────────────────────────────────
test-integration:
	$(PYTHON) -m pytest --run-integration -v tests/test_integration_live.py

# ── models ────────────────────────────────────────────────────────────────────
# Downloads Piper TTS voice models via git-lfs.
# The destination directory is resolved from vox_romana.yaml / config defaults.
# HuggingFace models (e.g. musicgen-large) are downloaded automatically on
# first use; see vox_romana/tts/music_engine.py.
models:
	@VOICES_DIR=$$($(PYTHON) -c \
	  "from vox_romana.core.paths import PIPER_VOICES_DIR; print(PIPER_VOICES_DIR)") && \
	  echo "Piper voices dir: $$VOICES_DIR" && \
	  if [ -d "$$VOICES_DIR" ] && [ -n "$$(ls -A "$$VOICES_DIR" 2>/dev/null)" ]; then \
	    echo "Piper voices already present — skipping clone."; \
	  else \
	    echo "Cloning rhasspy/piper-voices (requires git-lfs) …"; \
	    git lfs clone https://huggingface.co/rhasspy/piper-voices "$$VOICES_DIR"; \
	  fi
	@echo ""
	@echo "HF models are downloaded automatically on first use."
	@echo "To pre-fetch musicgen-large (~13.7 GB), run:"
	@echo "  $(PYTHON) -c \"from vox_romana.tts.music_engine import MusicGenEngine; MusicGenEngine()._ensure_loaded()\""

# ── clean ─────────────────────────────────────────────────────────────────────
# Removes the virtualenv, bytecode caches, and pytest artefacts.
# Does NOT touch output files, models, .env, or vox_romana.yaml.
clean:
	rm -rf .venv
	find . -type d -name __pycache__ -not -path './.git/*' -exec rm -rf {} +
	find . -type f -name '*.pyc'     -not -path './.git/*' -delete
	rm -rf .pytest_cache
	@echo "Clean complete. Run 'make install' to rebuild the environment."
