"""Knowledge LM — LoRA fine-tuning of a small causal language model.

Trains a LoRA adapter on (instruction, completion) pairs from the corpus
builder.  After training, ``generate(prompt)`` synthesizes from internalized
knowledge — the model "remembers" patterns across all studies, not just
retrieving individual records.

BASE MODEL:
    Default: ``Qwen/Qwen2-1.5B`` — trainable on M1 Max with MPS backend,
    small enough for fast iteration, large enough to synthesize meaningfully.
    Override via ``base_model_name``.

LoRA CONFIG:
    - Rank 16, alpha 32 — compact adapters (~8-16 MB).
    - Targets q_proj + v_proj attention layers.
    - Multiple adapters can coexist (e.g., per content_type).

TRAINING DATA FORMAT:
    Each example is formatted as:
        <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{completion}<|im_end|>

    This matches ChatML so the model learns the instruction-following pattern.

DEPENDENCY SPECS:
    torch>=2.4,<2.6                  (in requirements-data-deep.txt)
    transformers>=4.44,<4.46         (in requirements-data-deep.txt)
    peft>=0.12,<0.14                 (in requirements-data-deep.txt)
    datasets>=3.0,<3.2              (in requirements-data-deep.txt)

STORAGE:
    LoRA adapter saved to ``data/memory/lm/``.
    Base model is NOT saved locally — downloaded from HuggingFace on first use.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from vox_romana.core.paths import LM_DIR as DEFAULT_LM_DIR
DEFAULT_BASE_MODEL = "Qwen/Qwen2-1.5B"

# LoRA defaults.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# ChatML formatting.
_CHATML_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n{completion}<|im_end|>"
)


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ─────────────────────────────────────────────────────────────────────────────

def _load_training_deps():
    """Returns (AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, etc.)."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, PeftModel
    from datasets import Dataset
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "TrainingArguments": TrainingArguments,
        "Trainer": Trainer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "LoraConfig": LoraConfig,
        "get_peft_model": get_peft_model,
        "PeftModel": PeftModel,
        "Dataset": Dataset,
    }


def _load_inference_deps():
    """Lighter deps for inference only."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeLM
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeLM:
    """Small causal LM with LoRA adapter trained on your knowledge corpus.

    Args:
        adapter_dir: Directory containing (or to save) the LoRA adapter.
        base_model_name: HuggingFace model ID for the base model.
    """

    def __init__(
        self,
        *,
        adapter_dir: str | Path | None = None,
        base_model_name: str = DEFAULT_BASE_MODEL,
    ) -> None:
        self._adapter_dir = Path(adapter_dir or DEFAULT_LM_DIR)
        self._base_model_name = base_model_name
        self._model = None
        self._tokenizer = None

    @property
    def is_trained(self) -> bool:
        """True if a trained adapter exists on disk."""
        return (self._adapter_dir / "adapter_config.json").exists()

    def load(self) -> None:
        """Load the base model + LoRA adapter for inference."""
        deps = _load_inference_deps()
        torch = deps["torch"]

        # Determine device.
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        self._tokenizer = deps["AutoTokenizer"].from_pretrained(
            self._base_model_name, trust_remote_code=True,
        )

        base_model = deps["AutoModelForCausalLM"].from_pretrained(
            self._base_model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
            trust_remote_code=True,
        )

        if self.is_trained:
            self._model = deps["PeftModel"].from_pretrained(
                base_model, str(self._adapter_dir),
            )
            logger.info("Loaded LoRA adapter from %s", self._adapter_dir)
        else:
            self._model = base_model
            logger.info("Loaded base model (no adapter): %s", self._base_model_name)

        self._model.eval()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a completion from the knowledge LM.

        Args:
            prompt: The instruction/question to answer from memory.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Generated text (completion only, prompt stripped).
        """
        if self._model is None or self._tokenizer is None:
            self.load()

        # Format as ChatML.
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        deps = _load_inference_deps()
        torch = deps["torch"]

        inputs = self._tokenizer(formatted, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the new tokens.
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return result.strip()

    def train(
        self,
        examples_path: str | Path,
        *,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 512,
        gradient_accumulation_steps: int = 4,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = LORA_ALPHA,
    ) -> dict[str, Any]:
        """Fine-tune a LoRA adapter on (instruction, completion) examples.

        Args:
            examples_path: Path to lm_examples.jsonl from corpus builder.
            epochs: Training epochs.
            batch_size: Per-device batch size.
            learning_rate: Peak learning rate.
            max_length: Max sequence length (tokens).
            gradient_accumulation_steps: Gradient accumulation steps.
            lora_rank: LoRA rank (higher = more capacity, bigger adapter).
            lora_alpha: LoRA alpha scaling factor.

        Returns:
            Dict with training stats.
        """
        deps = _load_training_deps()
        torch = deps["torch"]

        # Determine device.
        if torch.backends.mps.is_available():
            device_type = "mps"
            fp16 = False  # MPS doesn't support fp16 training well
            bf16 = False
        elif torch.cuda.is_available():
            device_type = "cuda"
            fp16 = True
            bf16 = False
        else:
            device_type = "cpu"
            fp16 = False
            bf16 = False

        # Load tokenizer + base model.
        tokenizer = deps["AutoTokenizer"].from_pretrained(
            self._base_model_name, trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = deps["AutoModelForCausalLM"].from_pretrained(
            self._base_model_name,
            torch_dtype=torch.float32,  # Train in fp32 for stability
            trust_remote_code=True,
        )

        # Apply LoRA.
        lora_config = deps["LoraConfig"](
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = deps["get_peft_model"](model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "LoRA params: %d trainable / %d total (%.2f%%)",
            trainable, total, 100 * trainable / total,
        )

        # Load and format training data.
        examples_path = Path(examples_path)
        texts = []
        with open(examples_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                formatted = _CHATML_TEMPLATE.format(
                    instruction=ex["instruction"],
                    completion=ex["completion"],
                )
                texts.append(formatted)

        if not texts:
            logger.warning("No LM examples found in %s", examples_path)
            return {"examples_loaded": 0, "status": "skipped"}

        logger.info("Training LM on %d examples for %d epochs", len(texts), epochs)

        # Tokenize.
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        dataset = deps["Dataset"].from_dict({"text": texts})
        tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

        # Set up labels (same as input_ids for causal LM).
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples

        tokenized = tokenized.map(add_labels, batched=True)

        # Training arguments.
        output_dir = str(self._adapter_dir / "training_output")
        training_args = deps["TrainingArguments"](
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=10,
            save_strategy="epoch",
            fp16=fp16,
            bf16=bf16,
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Safer for MPS
        )

        collator = deps["DataCollatorForLanguageModeling"](
            tokenizer=tokenizer, mlm=False,
        )

        trainer = deps["Trainer"](
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=collator,
        )

        # Train.
        trainer.train()

        # Save LoRA adapter (not the full model).
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(self._adapter_dir))
        tokenizer.save_pretrained(str(self._adapter_dir))

        self._model = model
        self._tokenizer = tokenizer
        logger.info("LoRA adapter saved to %s", self._adapter_dir)

        return {
            "examples_loaded": len(texts),
            "epochs": epochs,
            "trainable_params": trainable,
            "total_params": total,
            "adapter_dir": str(self._adapter_dir),
            "status": "trained",
        }
