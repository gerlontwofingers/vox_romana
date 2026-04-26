# vox_romana

LLM-driven audio content generation. Feed it a brief, get a structured script and audio in any language.

## What this does

vox_romana uses LLM agents (currently DeepSeek) to generate scripted audio content — debates, news shows, podcasts, audiobooks — as structured JSON documents, then feeds those documents through a local text-to-speech pipeline to produce audio. The entire pipeline runs from a single CLI entry point.

```
prompt → classify → bootstrap → conduct → edit → JSON → translate → TTS → audio
```

## Architecture

```
vox_romana/
├── core/                   # Generic LLM infrastructure
│   ├── client.py           #   DeepSeek API client (OpenAI SDK wrapper)
│   ├── agent.py            #   Conversational agent with message history
│   ├── languages.py        #   Language registry, BCP-47 codes, normalize_lang_arg()
│   └── config.py           #   ContentConfig dataclass (language field)
├── shows/                  # Content-type implementations
│   ├── debates/            #   Debate generation pipeline
│   ├── news/               #   News show pipeline
│   ├── podcast/            #   Podcast / interview pipeline
│   └── audiobooks/         #   Long-form chapter narration pipeline
├── translate/              # Multi-language translation
│   ├── __init__.py         #   Translator ABC, TranslationError, re-exports
│   ├── llm_translator.py   #   LLMTranslator — JSON-array batch protocol
│   └── event_translator.py #   translate_show_data() — deep-copies and translates a show dict
├── tts/                    # Text-to-speech
│   ├── engine.py           #   BaseTTSEngine ABC (synthesize takes language= param)
│   ├── renderer.py         #   Language-aware renderer — picks voices, threads language
│   └── engines/
│       ├── kokoro_engine.py #  Kokoro (English-only; raises ValueError for other langs)
│       └── piper_engine.py  #  Piper (multilingual; scans installed ONNX models)
├── publish/                # Human-readable document output
│   ├── formatter.py        #   Formatters for debate/news/podcast/audiobook + i18n labels
│   ├── renderer.py         #   publish_document() with _{lang} suffix logic
│   └── models.py           #   FormattedDocument, PublishResult (both carry language field)
└── memory/                 # Local knowledge store
    └── ingest.py           #   DocumentIngester (PDF, DOCX, HTML, CSV, …)

run.py                      # Master CLI — coordinates all of the above
run_show.py                 # Generate script only (no audio render)
render_show.py              # Render audio from an existing script JSON
templates/
└── publish/
    └── _base.html.jinja2   # HTML template (uses {{ doc.language }} for <html lang>)
```

`run.py` is the recommended entry point — it chains script generation, TTS
render, and (optionally) document publishing. Use `run_show.py` /
`render_show.py` directly when you want to inspect or edit the JSON
between the two stages.

## Modes

| Mode | What it does |
|------|-------------|
| `full` (default) | Generate script → translate → render audio → publish document |
| `generate` | Text generation only — produces a JSON script |
| `render` | Audio rendering only from an existing JSON |
| `publish` | Format an existing JSON into TXT and/or PDF |
| `ingest` | Feed documents into the local knowledge base for `--research` |

## Installation

### 1. Prerequisites

**Python 3.12+** is required. Check with `python3 --version`.

**System packages** (install once, outside any virtualenv):

```bash
# macOS
brew install ffmpeg espeak-ng git-lfs
git lfs install

# Ubuntu / Debian
sudo apt install ffmpeg espeak-ng git-lfs
git lfs install
```

- `ffmpeg` — audio mixing and format conversion
- `espeak-ng` — phoneme library required by Piper TTS
- `git-lfs` — needed to pull the Piper voice model files

### 2. Clone the repo

```bash
git clone https://github.com/your-org/vox_romana.git
cd vox_romana
```

### 3. Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 4. API key

Get a key from [platform.deepseek.com](https://platform.deepseek.com), then:

```bash
cp .env.example .env             # if an example file exists, otherwise:
echo 'DEEPSEEK_API_KEY=sk-...' > .env
```

The key is only needed for `generate` and `full` modes. `render`, `publish`, and `ingest` are fully local.

### 5. Piper voice models

Piper uses ONNX voice model files checked in via git-lfs. If the repo includes them:

```bash
git lfs pull
```

Or set `PIPER_VOICES_DIR` in your `.env` to point at a directory of `.onnx` files you've downloaded separately.

### 6. Verify the install

```bash
python preflight.py
```

All required items should show `✓`. Optional items (`ollama`, Piper voices for non-English) can be added later.

### 7. (Optional) Configure

```bash
cp vox_romana.example.yaml vox_romana.yaml
# Edit vox_romana.yaml to set default engine, voice preferences, output paths, etc.
```

---

## Quick start

```bash
# Write a brief to a plain-text file (any path you like):
echo 'A 10-minute debate on universal basic income.' > my_brief.txt

# Generate + render:
python run.py my_brief.txt

# Publish as a formatted document:
python run.py --mode publish --script-input output/text/debates/my_brief.json
```

> **Note on examples:** Throughout this README you'll see paths like
> `prompts/ubi_debate.txt` — those are illustrative. The repo ships no
> sample prompts; create your own plain-text briefs and point the CLI at
> them. Output paths follow the pattern
> `output/text/{show_type}/{prompt_stem}_{tag}_{timestamp}.json`.

## Usage examples

### Full pipeline

```bash
# Generate + render (auto-detect type, Kokoro engine, mp3):
python run.py prompts/ubi_debate.txt

# With research, duration target, and custom audio path:
python run.py prompts/tech_news.txt --research --duration 15 \
    --audio-output output/audio/news/tech.mp3

# Publish a formatted TXT alongside the audio:
python run.py prompts/ubi_debate.txt --publish

# Full pipeline — English script, Spanish and French audio + docs:
python run.py prompts/ubi_debate.txt --out-lang en,es,fr --publish
```

### Generate only

```bash
python run.py prompts/ubi_debate.txt --mode generate
python run.py prompts/tech_news.txt --mode generate --segments 8 --tone urgent
```

### Render only

```bash
python run.py --mode render --script-input output/text/debates/ubi.json
python run.py --mode render --script-input output/text/news/tech.json \
    --engine piper --audio-output my_news.wav --format wav
```

### Publish only

```bash
# TXT (default):
python run.py --mode publish --script-input output/text/debates/ubi.json

# PDF + TXT:
python run.py --mode publish --script-input output/text/debates/ubi.json \
    --publish-format both --publish-output ~/Desktop

# All installed languages from a single JSON:
python run.py --mode publish --script-input output/text/debates/ubi.json \
    --out-lang all
```

### Ingest a knowledge base

```bash
python run.py --mode ingest --ingest-path papers/
python run.py --mode ingest --ingest-path report.pdf --ingest-tag research-2026
python run.py --mode ingest --ingest-path docs/ --dry-run
```

## Multi-language pipeline

vox_romana generates content once in the root language, then translates and renders audio + documents for each requested output language in a single command.

```
root JSON (en) ─┬─ render → debate.mp3     (English audio)
                │  publish → debate.txt    (English doc)
                ├─ translate → debate_es.json
                │  render → debate_es.mp3  (Spanish audio)
                │  publish → debate_es.txt (Spanish doc)
                └─ translate → debate_fr.json
                   render → debate_fr.mp3  (French audio)
                   publish → debate_fr.txt (French doc)
```

Output file names: root language uses the base name; non-root languages get a `_{lang}` suffix before the extension.

### Language flags

| Flag | Default | Description |
|------|---------|-------------|
| `--root-lang LANG` | `en` | Language the LLM generates content in |
| `--out-lang LANG[,LANG…]\|all` | *(root only)* | Comma-separated output languages, or `all` for every installed Piper voice |
| `--keep-translated` | off | Keep intermediate translated JSON files on disk instead of deleting after render/publish |
| `--translation-model MODEL` | *(same as generator)* | Override the LLM model used for translation passes |

### Examples

```bash
# Produce English + Spanish outputs:
python run.py prompts/ubi_debate.txt --out-lang en,es

# Produce every language that has a Piper voice installed:
python run.py prompts/ubi_debate.txt --out-lang all

# Generate in French, export Spanish and German too:
python run.py prompts/ubi_debate.txt --root-lang fr --out-lang fr,es,de

# Keep the translated JSONs for inspection:
python run.py prompts/ubi_debate.txt --out-lang es --keep-translated

# Use a faster/cheaper model for the translation pass:
python run.py prompts/ubi_debate.txt --out-lang es \
    --translation-model deepseek-chat
```

### TTS language support

> ⚠️ **Kokoro is English-only.** It raises a `ValueError` for any non-`en`
> language. If you pass `--out-lang es,fr,…` use either `--engine piper`
> or `--engine both` (the `MultiEngine`, which routes Kokoro for English
> and Piper for everything else). Forcing `--engine kokoro` with a
> non-English target language will abort the run.

- **Kokoro**: English only. Best quality on Apple Silicon (MLX-accelerated). Raises an error if asked for any other language.
- **Piper**: Multilingual. Scans `PIPER_VOICES_DIR` for installed ONNX models; available languages are those with at least one model file. Run `python run.py --out-lang all` to see what's installed.
- **Both** (`--engine both`): MultiEngine — uses Kokoro for English speakers and Piper for non-English speakers within the same show.

### Publish i18n

Section headings (ACT, Chapter, Segment) are localised for all output languages. Supported languages and their labels:

| Code | ACT | Chapter | Segment |
|------|-----|---------|---------|
| `en` | ACT | Chapter | Segment |
| `es` | ACTO | Capítulo | Segmento |
| `fr` | ACTE | Chapitre | Segment |
| `de` | AKT | Kapitel | Abschnitt |
| `pt` | ATO | Capítulo | Segmento |
| `it` | ATTO | Capitolo | Segmento |
| `nl` | BEDRIJF | Hoofdstuk | Segment |
| `pl` | AKT | Rozdział | Segment |
| `ru` | АКТ | Глава | Сегмент |
| `ja` | 幕 | 章 | セグメント |
| `zh` | 幕 | 章 | 部分 |
| `ko` | 막 | 장 | 세그먼트 |
| `ar` | فصل | فصل | مقطع |

Unknown language codes fall back to English labels.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes (generate/full) | Your DeepSeek API key |
| `GOOGLE_API_KEY` | No | Enables Google web search for `--research` |
| `GOOGLE_CSE_ID` | No | Custom Search Engine ID for `--research` |
| `PIPER_VOICES_DIR` | No | Override the directory scanned for Piper ONNX voice models |

Set these in a `.env` file at the project root (loaded automatically by `python-dotenv`).

### About `--research`

The `--research` flag turns on the bootstrap research stage: before the
conductor starts generating dialogue, the system gathers factual context
for the brief. Research uses a three-tier fallback:

1. **Local memory first** — the Chroma vector index + FTS5 store under
   `output/memory/` (populated via `--mode ingest`) is consulted.
2. **Google CSE** — if `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` are set,
   the system performs live web searches and caches the results.
3. **LLM fallback** — if no Google keys are configured, the LLM answers
   research questions from its own training data (less reliable; flagged
   in the run log).

The router auto-enables `--research` for briefs that look factual (news,
topical debates). Pass `--no-research` to force it off, or `--no-memory`
to skip the local knowledge base and force web-only lookups.

## Testing

```bash
# Unit tests only (no API calls, no local models required):
PYTHONPATH=. .venv/bin/pytest -q

# Include integration tests (requires DEEPSEEK_API_KEY in .env):
PYTHONPATH=. .venv/bin/pytest -q --run-integration

# Integration tests only:
PYTHONPATH=. .venv/bin/pytest -q --run-integration tests/test_translation_smoke.py
```

Integration tests (`@pytest.mark.integration`) make real DeepSeek API calls and are skipped by default. The TTS round-trip test additionally requires at least one non-English Piper ONNX voice installed.

## Dependencies

**System (install via your OS package manager — not pip):**
- `ffmpeg` — audio mixing backend (`brew install ffmpeg` / `apt install ffmpeg`)
- `espeak-ng` — phoneme library required by Piper TTS (`brew install espeak-ng` / `apt install espeak-ng`)
- `git-lfs` — required to pull Piper ONNX voice models (`brew install git-lfs` / `apt install git-lfs`)
- Optional: `pango cairo gdk-pixbuf libffi` — PDF publish output via WeasyPrint (`brew install pango cairo gdk-pixbuf libffi`)
- Optional: `ollama` — local LLM for university mode (`brew install ollama`)

**Python packages (`pip install -r requirements.txt`):**
- `openai` — DeepSeek API client (OpenAI SDK via `base_url` override)
- `python-dotenv` — `.env` loading
- `kokoro` — English TTS engine (local, no API key needed)
- `piper-tts` — Multilingual TTS engine (requires downloaded ONNX voice models)
- `pydub` — Audio mixing and format conversion
- Optional: `weasyprint` + `jinja2` — PDF publish output
- Optional: `chromadb` + `sentence-transformers` — Vector memory for `--research`
- Optional: `beautifulsoup4`, `python-docx`, `openpyxl` — Extra ingest formats

See `requirements.txt` for pinned versions.
