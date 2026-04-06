import json
import tempfile
import unittest
from pathlib import Path

from vox_romana import (
    DebateDocumentMerger,
    DebateMergeError,
    DebateParserRegistry,
    DebatePipeline,
    DebateSerializer,
    DebateTemplate01Parser,
    DebateTemplate02Parser,
    DebateValidator,
)


TEMPLATE_01_SAMPLE = """=== DEBATE_CHUNK_START ===
TEMPLATE_ID: template_01
DEBATE_ID: debate_alpha
DEBATE_TITLE: Should Cities Ban Cars Downtown?
DEBATE_TOPIC: Urban transit policy
DEBATE_FORMAT: public_forum
CHUNK_ORDER: 1
IS_FINAL_CHUNK: NO
SETTING: City hall
MODERATOR_MODE: SINGLE
MODERATOR_NAMES: Jordan Lee
AUDIENCE_PRESENT: YES
AUDIENCE_PARTICIPATION: NO
PARTICIPANT_NAMES: Riley Chen; Sam Ortiz
PARTICIPANT_COUNT: 2
NOTES_FOR_CONTINUITY: Sam should challenge the cost assumptions.

--- UTTERANCE START ---
GLOBAL_UTTERANCE_ORDER: 1
CHUNK_UTTERANCE_ORDER: 1
SPEAKER_ROLE: moderator
SPEAKER_NAME: Jordan Lee
TARGET: all
TONE: neutral
VOLUME: moderate
PACE: medium
PAUSE_BEFORE: none
PAUSE_AFTER: short
INTERRUPTION: NO
NONVERBAL: NONE
SPOKEN_TEXT: Welcome to tonight's debate.
--- UTTERANCE END ---

--- UTTERANCE START ---
GLOBAL_UTTERANCE_ORDER: 2
CHUNK_UTTERANCE_ORDER: 2
SPEAKER_ROLE: participant
SPEAKER_NAME: Riley Chen
TARGET: all
TONE: calm; analytical
VOLUME: moderate
PACE: medium
PAUSE_BEFORE: short
PAUSE_AFTER: short
INTERRUPTION: NO
NONVERBAL: NONE
SPOKEN_TEXT: Downtown car restrictions improve safety and air quality.
--- UTTERANCE END ---
=== DEBATE_CHUNK_END ===
"""


TEMPLATE_02_SAMPLE = """=== DEBATE_SEQUENCE_START ===
TEMPLATE_ID: template_02
DEBATE_ID: debate_beta
DEBATE_TITLE: Should Homework Be Optional?
DEBATE_TOPIC: Education policy
DEBATE_STYLE: crossfire
SEGMENT_ORDER: 1
IS_FINAL_SEGMENT: NO
SETTING: School auditorium
MODERATOR_MODE: NONE
MODERATOR_NAMES: NONE
AUDIENCE_PRESENT: YES
AUDIENCE_PARTICIPATION: YES
PARTICIPANT_NAMES: Ava Brooks; Leo Grant
PARTICIPANT_COUNT: 2
CARRY_FORWARD: Leo should push back on accountability.

+++ EVENT START +++
GLOBAL_EVENT_ORDER: 1
SEGMENT_EVENT_ORDER: 1
EVENT_KIND: speech
SPEAKER_ROLE: participant
SPEAKER_NAME: Ava Brooks
ADDRESSEE: all
TONE_SHIFT: firm
VOLUME: moderate
PACE: medium
PAUSE_BEFORE: none
PAUSE_AFTER: short
OVERLAP_WITH_PREVIOUS: NO
INTERRUPTS_PREVIOUS: NO
COMPLETION_STATUS: complete
NONVERBAL: NONE
TEXT: Optional homework supports individualized learning.
+++ EVENT END +++

+++ EVENT START +++
GLOBAL_EVENT_ORDER: 2
SEGMENT_EVENT_ORDER: 2
EVENT_KIND: interruption
SPEAKER_ROLE: participant
SPEAKER_NAME: Leo Grant
ADDRESSEE: Ava Brooks
TONE_SHIFT: urgent; confrontational
VOLUME: rising
PACE: fast
PAUSE_BEFORE: none
PAUSE_AFTER: short
OVERLAP_WITH_PREVIOUS: YES
INTERRUPTS_PREVIOUS: YES
COMPLETION_STATUS: fragment
NONVERBAL: crosstalk
TEXT: But that ignores students who need structure.
+++ EVENT END +++
=== DEBATE_SEQUENCE_END ===
"""


TEMPLATE_01_SAMPLE_CHUNK_2 = """=== DEBATE_CHUNK_START ===
TEMPLATE_ID: template_01
DEBATE_ID: debate_alpha
DEBATE_TITLE: Should Cities Ban Cars Downtown?
DEBATE_TOPIC: Urban transit policy
DEBATE_FORMAT: public_forum
CHUNK_ORDER: 2
IS_FINAL_CHUNK: YES
SETTING: City hall
MODERATOR_MODE: SINGLE
MODERATOR_NAMES: Jordan Lee
AUDIENCE_PRESENT: YES
AUDIENCE_PARTICIPATION: NO
PARTICIPANT_NAMES: Riley Chen; Sam Ortiz
PARTICIPANT_COUNT: 2
NOTES_FOR_CONTINUITY: NONE

--- UTTERANCE START ---
GLOBAL_UTTERANCE_ORDER: 3
CHUNK_UTTERANCE_ORDER: 1
SPEAKER_ROLE: participant
SPEAKER_NAME: Sam Ortiz
TARGET: Riley Chen
TONE: firm
VOLUME: moderate
PACE: medium
PAUSE_BEFORE: short
PAUSE_AFTER: short
INTERRUPTION: NO
NONVERBAL: NONE
SPOKEN_TEXT: The funding model still looks unrealistic.
--- UTTERANCE END ---

--- UTTERANCE START ---
GLOBAL_UTTERANCE_ORDER: 4
CHUNK_UTTERANCE_ORDER: 2
SPEAKER_ROLE: moderator
SPEAKER_NAME: Jordan Lee
TARGET: all
TONE: neutral
VOLUME: moderate
PACE: medium
PAUSE_BEFORE: short
PAUSE_AFTER: long
INTERRUPTION: NO
NONVERBAL: applause
SPOKEN_TEXT: That concludes this portion of the discussion.
--- UTTERANCE END ---
=== DEBATE_CHUNK_END ===
"""


class DebateTemplateParserTests(unittest.TestCase):
    def test_template_01_parser_normalizes_chunk(self) -> None:
        parsed = DebateTemplate01Parser().parse(TEMPLATE_01_SAMPLE)

        self.assertEqual(parsed.template_id, "template_01")
        self.assertEqual(len(parsed.chunks), 1)
        self.assertEqual(parsed.chunks[0].title, "Should Cities Ban Cars Downtown?")
        self.assertEqual(parsed.chunks[0].participants, ("Riley Chen", "Sam Ortiz"))
        self.assertEqual(parsed.chunks[0].events[1].tone, ("calm", "analytical"))

    def test_template_02_parser_normalizes_dynamic_events(self) -> None:
        parsed = DebateTemplate02Parser().parse(TEMPLATE_02_SAMPLE)

        self.assertEqual(parsed.template_id, "template_02")
        self.assertEqual(parsed.chunks[0].chunk_order, 1)
        self.assertEqual(parsed.chunks[0].debate_format, "crossfire")
        self.assertTrue(parsed.chunks[0].events[1].interruption)
        self.assertTrue(parsed.chunks[0].events[1].overlap_with_previous)

    def test_registry_detects_parser_from_template_id(self) -> None:
        registry = DebateParserRegistry()

        parsed_01 = registry.parse(TEMPLATE_01_SAMPLE)
        parsed_02 = registry.parse(TEMPLATE_02_SAMPLE)

        self.assertEqual(parsed_01.parser_name, "DebateTemplate01Parser")
        self.assertEqual(parsed_02.parser_name, "DebateTemplate02Parser")

    def test_merger_orders_chunks_and_events(self) -> None:
        registry = DebateParserRegistry()
        merger = DebateDocumentMerger()

        merged = merger.merge_documents(
            (
                registry.parse(TEMPLATE_01_SAMPLE_CHUNK_2),
                registry.parse(TEMPLATE_01_SAMPLE),
            )
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].debate_id, "debate_alpha")
        self.assertEqual(tuple(chunk.chunk_order for chunk in merged[0].chunks), (1, 2))
        self.assertEqual(tuple(event.global_order for event in merged[0].events), (1, 2, 3, 4))
        self.assertEqual(merged[0].continuity_notes, (
            "Sam should challenge the cost assumptions.",
        ))

    def test_registry_can_parse_and_merge_many_inputs(self) -> None:
        registry = DebateParserRegistry()

        merged = registry.parse_and_merge((TEMPLATE_01_SAMPLE, TEMPLATE_01_SAMPLE_CHUNK_2))

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].participants, ("Riley Chen", "Sam Ortiz"))
        self.assertEqual(merged[0].source_count, 2)

    def test_merger_rejects_conflicting_metadata(self) -> None:
        registry = DebateParserRegistry()
        conflict_text = TEMPLATE_01_SAMPLE_CHUNK_2.replace(
            "DEBATE_TITLE: Should Cities Ban Cars Downtown?",
            "DEBATE_TITLE: Should Downtown Streets Stay Open to Cars?",
        )

        with self.assertRaises(DebateMergeError):
            DebateDocumentMerger().merge_documents(
                (registry.parse(TEMPLATE_01_SAMPLE), registry.parse(conflict_text))
            )


# ── Validator tests ────────────────────────────────────────────────────────

class DebateValidatorTests(unittest.TestCase):
    def test_valid_document_passes(self) -> None:
        parsed = DebateTemplate01Parser().parse(TEMPLATE_01_SAMPLE)
        report = DebateValidator().validate(parsed)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.errors, [])

    def test_unknown_tone_produces_warning(self) -> None:
        # Inject a bogus tone value into the raw text
        bad_text = TEMPLATE_01_SAMPLE.replace("TONE: neutral", "TONE: sarcastic")
        parsed = DebateTemplate01Parser().parse(bad_text)
        report = DebateValidator().validate(parsed)

        # Still valid (warnings don't block), but issue is recorded
        self.assertTrue(report.is_valid)
        tone_warnings = [i for i in report.warnings if i.field_name == "tone"]
        self.assertGreater(len(tone_warnings), 0)

    def test_unknown_speaker_produces_warning(self) -> None:
        bad_text = TEMPLATE_01_SAMPLE.replace(
            "SPEAKER_NAME: Riley Chen",
            "SPEAKER_NAME: Ghost Speaker",
        )
        parsed = DebateTemplate01Parser().parse(bad_text)
        report = DebateValidator().validate(parsed)

        name_warnings = [i for i in report.warnings if i.field_name == "speaker_name"]
        self.assertGreater(len(name_warnings), 0)

    def test_duplicate_global_orders_produce_error(self) -> None:
        # Make both utterances share global order 1
        bad_text = TEMPLATE_01_SAMPLE.replace(
            "GLOBAL_UTTERANCE_ORDER: 2",
            "GLOBAL_UTTERANCE_ORDER: 1",
        )
        parsed = DebateTemplate01Parser().parse(bad_text)
        report = DebateValidator().validate(parsed)

        self.assertFalse(report.is_valid)
        order_errors = [i for i in report.errors if i.field_name == "global_order"]
        self.assertGreater(len(order_errors), 0)

    def test_summary_is_readable(self) -> None:
        parsed = DebateTemplate01Parser().parse(TEMPLATE_01_SAMPLE)
        report = DebateValidator().validate(parsed)
        summary = report.summary()

        self.assertIn("debate_alpha", summary)
        self.assertIn("0 error(s)", summary)


# ── Serializer tests ──────────────────────────────────────────────────────

class DebateSerializerTests(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        registry = DebateParserRegistry()
        merger = DebateDocumentMerger()
        serializer = DebateSerializer()

        parsed = registry.parse(TEMPLATE_01_SAMPLE)
        merged = merger.merge_document(parsed)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "debate.json"
            serializer.save_merged(merged, out_path)

            loaded = serializer.load(out_path)

        self.assertEqual(loaded["debate_id"], "debate_alpha")
        self.assertEqual(len(loaded["events"]), 2)

    def test_to_json_string_is_valid_json(self) -> None:
        parsed = DebateTemplate01Parser().parse(TEMPLATE_01_SAMPLE)
        serializer = DebateSerializer()

        json_str = serializer.to_json_string(parsed)
        data = json.loads(json_str)

        self.assertEqual(data["template_id"], "template_01")


# ── Pipeline tests ─────────────────────────────────────────────────────────

class DebatePipelineTests(unittest.TestCase):
    def test_full_pipeline_happy_path(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run([TEMPLATE_01_SAMPLE, TEMPLATE_01_SAMPLE_CHUNK_2])

        self.assertTrue(result.ok)
        self.assertEqual(len(result.parsed), 2)
        self.assertEqual(len(result.merged), 1)
        self.assertEqual(result.merged[0].debate_id, "debate_alpha")
        self.assertEqual(len(result.merged[0].events), 4)

    def test_pipeline_captures_parse_failures(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run(["this is not a debate template at all"])

        self.assertFalse(result.ok)
        self.assertEqual(len(result.parse_failures), 1)
        self.assertEqual(len(result.merged), 0)

    def test_pipeline_mixed_good_and_bad(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run([TEMPLATE_01_SAMPLE, "garbage text"])

        # One parsed, one failed
        self.assertEqual(len(result.parsed), 1)
        self.assertEqual(len(result.parse_failures), 1)
        # The good one still merges
        self.assertEqual(len(result.merged), 1)

    def test_pipeline_strict_mode_blocks_warnings(self) -> None:
        # Inject a bad tone so there's a warning
        bad_text = TEMPLATE_01_SAMPLE.replace("TONE: neutral", "TONE: sarcastic")
        pipeline = DebatePipeline()

        # Non-strict: warning is fine, still merges
        result_lenient = pipeline.run([bad_text])
        self.assertEqual(len(result_lenient.merged), 1)

        # Strict: warning blocks merge
        result_strict = pipeline.run([bad_text], strict=True)
        self.assertEqual(len(result_strict.merged), 0)

    def test_pipeline_saves_to_directory(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run([TEMPLATE_01_SAMPLE, TEMPLATE_01_SAMPLE_CHUNK_2])

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = pipeline.save_all(result, tmpdir)

            self.assertEqual(len(paths), 1)
            self.assertTrue(paths[0].exists())
            self.assertEqual(paths[0].name, "debate_alpha.json")

            # Verify the file is valid JSON with the right content
            data = json.loads(paths[0].read_text())
            self.assertEqual(data["debate_id"], "debate_alpha")
            self.assertEqual(len(data["events"]), 4)

    def test_pipeline_to_json_returns_string(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run([TEMPLATE_02_SAMPLE])

        json_str = pipeline.to_json(result.merged[0])
        data = json.loads(json_str)

        self.assertEqual(data["debate_id"], "debate_beta")

    def test_pipeline_summary_is_useful(self) -> None:
        pipeline = DebatePipeline()
        result = pipeline.run([TEMPLATE_01_SAMPLE, "bad input"])

        summary = result.summary()
        self.assertIn("Parsed: 1", summary)
        self.assertIn("Parse failures: 1", summary)
        self.assertIn("PARSE FAIL", summary)


if __name__ == "__main__":
    unittest.main()