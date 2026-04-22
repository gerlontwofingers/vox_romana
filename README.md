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
│   └── news/               #   News show pipeline
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
templates/
└── publish/
    └── _base.html.jinja2   # HTML template (uses {{ doc.language }} for <html lang>)
```

## Modes

| Mode | What it does |
|------|-------------|
| `full` (default) | Generate script → translate → render audio → publish document |
| `generate` | Text generation only — produces a JSON script |
| `render` | Audio rendering only from an existing JSON |
| `publish` | Format an existing JSON into TXT and/or PDF |
| `ingest` | Feed documents into the local knowledge base for `--research` |

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Add your DeepSeek API key to .env
echo 'DEEPSEEK_API_KEY=sk-...' >> .env

# Generate + render a debate:
python run.py prompts/ubi_debate.txt

# Publish as a formatted document:
python run.py --mode publish --script-input output/text/debates/ubi.json
```

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

- **Kokoro**: English only. Raises an error if asked for any other language — use Piper for non-English output.
- **Piper**: Multilingual. Scans the `PIPER_VOICES_DIR` for installed ONNX models; available languages are those with at least one model file. Run `python run.py --out-lang all` to see what's installed.

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
| `GOOGLE_API_KEY` | No | Enables `--research` web search |
| `GOOGLE_CSE_ID` | No | Custom Search Engine ID for `--research` |

Set these in a `.env` file at the project root (loaded automatically by `python-dotenv`).

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

- Python 3.12+
- `openai` — DeepSeek API client (OpenAI SDK via `base_url` override)
- `python-dotenv` — `.env` loading
- `kokoro` — English TTS engine (local, no API key needed)
- `piper-tts` — Multilingual TTS engine (requires downloaded ONNX voice models)
- `pydub` — Audio mixing and format conversion
- Optional: `weasyprint` + `jinja2` — PDF publish output
- Optional: `chromadb` + `sentence-transformers` — Vector memory for `--research`
- Optional: `beautifulsoup4`, `python-docx`, `openpyxl` — Extra ingest formats

See `requirements.txt` for pinned versions.
