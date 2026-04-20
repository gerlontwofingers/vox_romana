# vox_romana

LLM-driven audio content generation. Feed it a brief, get a structured script, pipe it to text-to-speech.

## What this does

vox_romana uses LLM agents (currently DeepSeek) to generate scripted audio content — debates, news shows, audiobooks — as structured, machine-readable documents. Those documents then feed into a text-to-speech pipeline to produce actual audio.

The system is designed around two core ideas:

1. **LLMs generate plain-text scripts** following strict templates. Plain text is more durable than JSON under LLM context pressure — models break JSON structure when they're working hard on content.
2. **Python parses, validates, merges, and serializes** those scripts into clean JSON for downstream consumption (TTS, editing tools, analysis).

## Architecture

```
vox_romana/
├── core/               # Generic LLM infrastructure
│   ├── client.py       #   DeepSeek API client (OpenAI SDK wrapper)
│   ├── agent.py        #   Conversational agent with message history
│   └── models.py       #   AgentResponse dataclass
├── shows/              # Content-type implementations
│   └── debates/        #   Debate generation pipeline
│       ├── models.py   #     DebateEvent, DebateChunk, MergedDebate
│       ├── parsers.py  #     Template parsers + registry
│       ├── validator.py#     Structural + value validation
│       ├── merge.py    #     Multi-chunk merging with conflict detection
│       ├── serializer.py#    JSON read/write
│       └── pipeline.py #     End-to-end: parse → validate → merge → serialize
└── __init__.py         # Re-exports everything for convenience

templates/              # Plain-text templates for LLM generation
└── debates/
    ├── template_01.txt #   Structured format (chunks + utterances)
    └── template_02.txt #   Dynamic format (sequences + events, interruptions, overlap)

tests/
└── test_debate_parsers.py  # 20 tests covering all pipeline layers
```

## Content types

| Type | Status | Generation pattern |
|------|--------|--------------------|
| **Debates** | Working pipeline (parse → validate → merge → serialize) | Reactive: multi-agent turn loop |
| **News shows** | Planned | Hybrid: scripted anchoring + reactive panel |
| **Audiobooks** | Planned | Scripted: sequential chapter rendering |

## How the debate pipeline works

```python
from vox_romana import DebatePipeline

pipeline = DebatePipeline()

# Feed it raw text chunks from the LLM (one string per chunk)
result = pipeline.run([chunk_1_text, chunk_2_text])

# Check what happened
print(result.summary())

# Save merged debates to JSON
if result.ok:
    pipeline.save_all(result, "output/")
```

The pipeline:
1. **Parses** each raw text chunk, auto-detecting the template format
2. **Validates** structural integrity and value correctness (warnings vs errors)
3. **Merges** chunks sharing the same debate ID into unified documents
4. **Serializes** to pretty-printed JSON

## Where this is headed

The current pipeline handles the *output side* — parsing and validating what an LLM has already generated. The next phase builds the *generation side*:

- **Multi-agent conductor**: One DeepSeek agent per participant (moderator, debaters, audience). A Director agent sets mood/arc every 5–10 turns. The turn loop is strictly sequential — agent B depends on agent A's output.
- **Bootstrap from brief**: One natural-language description → automatic config generation via a chain of focused LLM calls (Architect → Profile Builder → Director Brief). ~4–7 API calls, ~$0.01.
- **Personality system**: Voice fingerprints (rhetorical habits, verbal tics), behavioral triggers, emotional arcs, optional reference corpus for real public figures.
- **Two generation patterns**: Reactive (multi-agent turn loop for debates, interviews, panels) and Scripted (sequential rendering for audiobooks, narrated docs, news reads). Both produce the same Event/Segment objects → same pipeline → same TTS.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Set your DeepSeek API key
export DEEPSEEK_API_KEY="your-key-here"
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Dependencies

- Python 3.12+
- `openai==2.30.0` (used as the DeepSeek client via base_url override)
