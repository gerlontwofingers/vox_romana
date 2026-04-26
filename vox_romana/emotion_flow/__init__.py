"""vox_romana.emotion_flow — Cross-Lingual Emotion Flow Analysis.

Public API:
    models          — all dataclasses
    reproducibility — freeze_randomness, build_manifest
    loader          — FlexibleLoader
    translator      — TranslatorAgent
    tagger          — EmotionTaggerAgent
    transition      — TransitionBuilderAgent
    semantic        — SemanticFlowAgent
    tester          — NullHypothesisTesterAgent
    visualizer      — EmotionFlowVisualizer
    reporter        — EmotionFlowReporter
    show_adapter    — EmotionFlowShowAdapter
"""

from .loader import FlexibleLoader
from .tagger import EmotionTaggerAgent
from .transition import TransitionBuilderAgent
from .semantic import SemanticFlowAgent
from .tester import NullHypothesisTesterAgent
from .visualizer import EmotionFlowVisualizer
from .reporter import EmotionFlowReporter
from .pipeline import EmotionFlowPipeline
from .models import (
    EMOTION_LABELS,
    EmotionFlowReport,
    EmotionSequence,
    EmotionTag,
    NullTestResult,
    ReproducibilityManifest,
    SemanticFlowResult,
    SpeechSegment,
    TransitionMatrix,
)
from .reproducibility import build_manifest, freeze_randomness
from .show_adapter import EmotionFlowShowAdapter
from .translator import TranslatorAgent

__all__ = [
    # models
    "EMOTION_LABELS",
    "SpeechSegment",
    "EmotionTag",
    "EmotionSequence",
    "TransitionMatrix",
    "SemanticFlowResult",
    "NullTestResult",
    "ReproducibilityManifest",
    "EmotionFlowReport",
    # reproducibility
    "freeze_randomness",
    "build_manifest",
    # loader
    "FlexibleLoader",
    # translator
    "TranslatorAgent",
    # tagger
    "EmotionTaggerAgent",
    # transition
    "TransitionBuilderAgent",
    # semantic
    "SemanticFlowAgent",
    # tester
    "NullHypothesisTesterAgent",
    # visualizer
    "EmotionFlowVisualizer",
    # reporter
    "EmotionFlowReporter",
    # pipeline
    "EmotionFlowPipeline",
    # show adapter
    "EmotionFlowShowAdapter",
]
