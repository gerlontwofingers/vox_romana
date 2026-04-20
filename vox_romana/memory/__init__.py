"""vox_romana.memory — learned semantic memory from the knowledge catalog.

Two complementary memory systems that turn the catalog's episodic records
into associative, generative recall:

    encoder  — Sentence-transformer fine-tuned on your corpus.  Produces
               embeddings for semantic search (replaces FTS5 in recall).
    lm       — Small causal LM (LoRA adapter) that synthesizes what it
               "remembers" from internalized knowledge.

Both are OPTIONAL.  If models aren't trained yet, the recall interface
degrades gracefully to the catalog's FTS5 full-text search.

DEPENDENCY SPECS (for human review):
    requirements-data-deep.txt — torch, sentence-transformers, transformers,
                                  peft (LoRA), datasets (HuggingFace)

Structure:
    corpus.py   — Builds training data from catalog + ingested documents.
    ingest.py   — PDF/text extraction, chunking, metadata tagging.
    encoder.py  — Contrastive fine-tuning of sentence-transformer.
    lm.py       — LoRA fine-tuning of small causal language model.
    recall.py   — Unified query interface: semantic search + LM synthesis.
    trainer.py  — Orchestrates corpus build → encoder → LM training.

STORAGE:
    data/memory/corpus/       — Generated training data (JSONL files).
    data/memory/encoder/      — Sentence-transformer weights.
    data/memory/lm/           — LoRA adapter weights.
    data/memory/documents/    — Ingested user documents (chunked).
"""

from .recall import MemoryRecall

__all__ = ["MemoryRecall"]
