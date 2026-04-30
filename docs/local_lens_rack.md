# Local Lens Rack

The **lens rack** is vox_romana's central abstraction for picking *which*
LLM is used for *which* job.  It treats local and cloud models as a set
of named **lenses**, each with creative metadata (`personality_hint`,
`cost_tier`, `language_strengths`).  A weaker or weirder model is not a
fallback — it's the right voice for some characters.

> A small, brittle model that mangles three languages per sentence is
> not a bug; it's Salvatore from *The Name of the Rose*.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                       LLMLensRack (registry)                       │
│                                                                    │
│   role "default"          ──┐                                      │
│   role "translator"       ──┼──► routes ──► lens name ──► LLMLens │
│   role "character"        ──┤                              │       │
│   role "character:Sal..." ──┘                              ▼       │
│                                                       (lazy build) │
│                                                            │       │
│                                  ┌─────────────────────────┴────┐  │
│                                  ▼                              ▼  │
│                       DeepSeekCloudClient              OllamaClient │
│                       (api.deepseek.com)         (localhost:11434)  │
└────────────────────────────────────────────────────────────────────┘
```

* `LLMLens` — named `LLMClient` with metadata.  Built lazily via a
  factory; the underlying client is cached after first use.
* `LLMLensRack.select(role=…, character=…)` — picks the most specific
  matching lens.  Precedence: `character:<exact>` > `character`
  catch-all > role > `default`.

## Built-in lens catalog

| Name                    | Backend           | Model            | Tier  | Personality hint                                                                                  |
|-------------------------|-------------------|------------------|-------|---------------------------------------------------------------------------------------------------|
| `cloud-deepseek`        | DeepSeek          | `deepseek-chat`  | cloud | Smart, even, neutral.  The default workhorse.                                                     |
| `local-r1-distill`      | Ollama            | `deepseek-r1:7b` | local | Strong reasoning for a 7B; reflective and a touch verbose.                                        |
| `local-qwen-polyglot`   | Ollama            | `qwen2.5:7b`     | local | Best multilingual coverage of the locals.  Translator default candidate.                          |
| `local-llama-dim`       | Ollama            | `llama3.2:3b`    | local | Small, brittle, sometimes naive.  Use for innocents, drunks, code-switching monks — the Salvatore lens. |

The cloud lens is constructed eagerly; the locals are constructed
lazily so a user who never opts into local inference does not need
Ollama installed.

## Setup

```bash
brew install ollama
brew services start ollama          # or: OLLAMA_FLASH_ATTENTION=1 ollama serve

ollama pull deepseek-r1:7b          # 4.7 GB — reasoning lens
ollama pull qwen2.5:7b              # 4.7 GB — polyglot lens
ollama pull llama3.2:3b             # 2.0 GB — Salvatore lens
```

Health-check:

```bash
curl -s http://localhost:11434/api/tags | jq '.models[].name'
```

## Casting a character to a lens

The headline use-case is voicing a specific character with a specific
lens.  Two ways:

### Via YAML config

```yaml
# vox_romana.yaml
llm:
  default: cloud-deepseek
  roles:
    translator: local-qwen-polyglot
    "character:Salvatore": local-llama-dim
```

### Via CLI override

```bash
python run.py generate \
    --llm-lens character:Salvatore=local-llama-dim \
    --llm-lens translator=local-qwen-polyglot
```

### Via the casting probe

Listen to every lens read the *same* monologue before you commit:

```bash
python scripts/lens_character_probe.py \
    --character "Salvatore: polyglot lay brother in a 14th-c. abbey, \
mangles Latin/Italian/Provençal in every sentence, fearful, devout" \
    --prompt "Describe what you saw last night in the scriptorium." \
    --out reports/salvatore.md
```

The report has one section per lens with personality-hint, latency,
and the verbatim response.

## Parity check before reassigning a role

Before reassigning a structured role (e.g. `conductor`) from cloud to
local, run the parity check to make sure the local lens still produces
valid JSON:

```bash
python scripts/lens_parity.py --out reports/parity.md
```

The report has a checkmark grid: ✓ valid JSON, ~ invalid JSON, ✗
exception.  Roles that score ✓ across the built-in prompts are safe to
reassign.

## Backend override

`run.py generate --llm-backend local` swaps the **default** lens from
`cloud-deepseek` to `local-r1-distill` for the run.  Per-role
assignments still win.  Use `--llm-backend cloud` to force the cloud
default; `auto` (the default) uses whatever the YAML says.

## Ollama-specific niceties

`OllamaClient` is a `DeepSeekClient` subclass, so every existing
call-site works unchanged.  In addition:

* `chat(..., response_format={"type": "json_object"})` is translated
  into Ollama's `extra_body={"format": "json"}` so JSON-mode prompts
  port over without code changes.
* `is_alive()` pings `GET /api/tags` for a fast health check.
* `available_models()` returns the list of locally installed tags so
  the lens probe can degrade gracefully when a model is missing.
