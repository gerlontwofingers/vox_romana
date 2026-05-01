# Local stack — running vox_romana fully offline

This document describes the local inference stack that powers vox_romana
when no cloud keys are available (or when cost/privacy matters), together
with the MCP server that exposes the full pipeline to AI coding tools like
Cursor and VS Code.

---

## Architecture overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  VS Code / Cursor / Claude Desktop                                    │
│  (MCP client — connects over stdio)                                   │
└──────────────────────┬───────────────────────────────────────────────┘
                       │  MCP/stdio
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  vox_romana MCP server  (python -m vox_romana.mcp)                   │
│                                                                       │
│  Read-only tools          │  Write / action tools                     │
│  ─────────────────────    │  ────────────────────────────────────     │
│  vox_search_memory        │  vox_ingest_document                      │
│  vox_list_episodes        │  vox_kick_off_generate   ──► _jobs dict   │
│  vox_get_episode          │  vox_get_job_status      ◄── poll         │
│  vox_search_corpus        │  vox_render_existing                      │
│  vox_list_voices          │  vox_publish_existing                     │
│  vox_list_lenses          │  vox_probe_character                      │
└──────────────────────┬────┴───────────────────────────────────────────┘
                       │
          ┌────────────┼─────────────────────────────┐
          ▼            ▼                             ▼
   ┌─────────┐  ┌────────────┐              ┌────────────────┐
   │MemoryDB │  │ ChromaDB   │              │  LLMLensRack   │
   │(SQLite) │  │(vector idx)│              │  (lens router) │
   └─────────┘  └────────────┘              └───────┬────────┘
                                                    │
                             ┌──────────────────────┼──────────────┐
                             ▼                      ▼              ▼
                      ┌────────────┐  ┌─────────────────┐  ┌──────────────┐
                      │ DeepSeek   │  │  Ollama          │  │  Ollama      │
                      │ Cloud API  │  │  deepseek-r1:7b  │  │  qwen2.5:7b  │
                      │(cloud tier)│  │  (local tier)    │  │ (polyglot)   │
                      └────────────┘  └─────────────────┘  └──────────────┘
```

---

## LLM lens catalog

| Lens name            | Backend | Model             | Cost tier | Notes                              |
|----------------------|---------|-------------------|-----------|------------------------------------|
| `cloud-deepseek`     | cloud   | deepseek-chat     | cloud     | Sharpest; use for conductor turns  |
| `local-r1-distill`   | local   | deepseek-r1:7b    | local     | Local reasoning; good for debate   |
| `local-qwen-poly`    | local   | qwen2.5:7b        | local     | Best multilingual coverage         |
| `local-llama-dim`    | local   | llama3.2:3b       | local     | Fastest; best for one-liner roles  |

### Casting recipe

```python
from vox_romana.core.lens import build_default_rack

rack = build_default_rack()

# Audition character before production
# (or use the vox_probe_character MCP tool directly from Cursor)
```

---

## MCP server

### Starting

```bash
# Install server (already in requirements.txt)
.venv/bin/pip install mcp

# Start over stdio (invoked automatically by Claude Desktop / VS Code)
.venv/bin/python -m vox_romana.mcp

# Test by listing tools
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | .venv/bin/python -m vox_romana.mcp
```

### VS Code wiring

```bash
cp .vscode/mcp.json.example .vscode/mcp.json
# Cmd+Shift+P → Developer: Reload Window
```

The bundled example uses `${workspaceFolder}/.venv/bin/python` as the
command — this is required so the server process finds `mcp` and all
vox_romana packages.  Bare `python` picks up the system interpreter and
will fail with `ModuleNotFoundError`.  On Windows substitute
`.venv\Scripts\python.exe`.

After reload, open Copilot Chat and click the **tools icon** (hammer).
You should see the **vox_romana** server listed with its 12 tools.

### Claude Desktop wiring

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vox_romana": {
      "command": "/path/to/vox_romana/.venv/bin/python",
      "args": ["-m", "vox_romana.mcp"],
      "cwd": "/path/to/vox_romana"
    }
  }
}
```

Replace `/path/to/vox_romana` with the absolute path to your clone.

---

## Tool reference

### Read-only tools

| Tool | Description |
|------|-------------|
| `vox_search_memory` | Semantic search over the knowledge store + conversation memory |
| `vox_list_episodes` | List recent shows from the catalog |
| `vox_get_episode` | Fetch one episode's metadata by run-id |
| `vox_search_corpus` | Vector similarity search over ingested documents |
| `vox_list_voices` | Enumerate Kokoro + Piper voice IDs |
| `vox_list_lenses` | Show the LLM lens registry + current routes |

### Write / action tools

| Tool | Destructive | Description |
|------|-------------|-------------|
| `vox_ingest_document` | No | Add a file to the knowledge store |
| `vox_kick_off_generate` | Yes | Spawn a generate job; returns `job_id` |
| `vox_get_job_status` | No | Poll a running generate job |
| `vox_render_existing` | No | Synthesise audio from an existing show JSON |
| `vox_publish_existing` | No | Export a show JSON as TXT / PDF |
| `vox_probe_character` | No | Audition a character across all lenses |

---

## Ollama setup

```bash
# Install (macOS)
brew install ollama
brew services start ollama

# Pull the three local models (one-time, ~11 GB total)
ollama pull deepseek-r1:7b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b

# Verify
ollama list
```

Health-check from Python:

```python
from vox_romana.local.ollama import OllamaClient
client = OllamaClient(model="deepseek-r1:7b", base_url="http://localhost:11434/v1")
print(client.is_alive())        # True
print(client.available_models()) # ['deepseek-r1:7b', 'qwen2.5:7b', 'llama3.2:3b']
```

---

## Job registry

`vox_kick_off_generate` runs `run.py --mode generate` in a background
thread and tracks it in an in-process dict (`_JOBS`).  The MCP server
process is long-lived (it stays alive as long as the editor is open),
so jobs survive individual tool calls.

Jobs expire when the MCP server process exits.  For durable tracking,
inspect `output/text/` — completed shows always write their JSON there.

---

## Running the full local stack offline

```bash
# 1. Start Ollama (if not already running as a service)
brew services start ollama

# 2. Generate a show entirely locally
python run.py "debate on whether cats are better than dogs" \
  --type debate \
  --llm-backend local \
  --no-research

# 3. Or via MCP from Cursor:
#    vox_kick_off_generate("debate on cats vs dogs", show_type="debate",
#                          llm_backend="local", no_research=True)
```

---

## Security note

The MCP server launches subprocesses (`run.py`) with the arguments
supplied by the MCP client.  Prompt injection via malicious `prompt_text`
values is mitigated by:

1. The text is written to a named temp file (never passed on the command
   line), so shell metacharacters cannot escape.
2. Subprocess is invoked via `subprocess.run(list, ...)` — no shell
   interpolation.
3. Only `run.py` in the repo root is invoked; no user-controlled binary.

Do **not** expose the MCP server over the network without authentication.
