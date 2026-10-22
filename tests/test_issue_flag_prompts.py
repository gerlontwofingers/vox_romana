"""Every cast system prompt must tell the LLM the optional issue_flag field exists.

The capture side (BaseConductor._capture_issue_flag) is tested per-conductor;
these tests cover the other half — that agents are actually told they can
use it. Without this, the capture path is dead code.
"""

from __future__ import annotations

import string

import pytest

from vox_romana.core.messages import ISSUE_FLAG_PROMPT_NOTE
from vox_romana.shows.audiobooks import prompts as audiobook_prompts
from vox_romana.shows.debates import prompts as debate_prompts
from vox_romana.shows.lecture import prompts as lecture_prompts
from vox_romana.shows.news import prompts as news_prompts
from vox_romana.shows.podcast import prompts as podcast_prompts

CAST_PROMPTS = [
    ("debate participant", debate_prompts.PARTICIPANT_SYSTEM_PROMPT),
    ("debate moderator", debate_prompts.MODERATOR_SYSTEM_PROMPT),
    ("debate show host", debate_prompts.SHOW_HOST_SYSTEM_PROMPT),
    ("news anchor", news_prompts.ANCHOR_SYSTEM_PROMPT),
    ("news show host", news_prompts.SHOW_HOST_SYSTEM_PROMPT),
    ("podcast hosts", podcast_prompts.PODCAST_HOST_SYSTEM_PROMPT),
    ("audiobook narrator", audiobook_prompts.NARRATOR_SYSTEM_PROMPT),
    ("lecture professor", lecture_prompts.PROFESSOR_SYSTEM_PROMPT),
]


@pytest.mark.parametrize("label,prompt", CAST_PROMPTS, ids=[c[0] for c in CAST_PROMPTS])
def test_cast_prompt_advertises_issue_flag(label, prompt):
    assert '"issue_flag"' in prompt, f"{label} prompt never mentions issue_flag"


@pytest.mark.parametrize("label,prompt", CAST_PROMPTS, ids=[c[0] for c in CAST_PROMPTS])
def test_cast_prompt_issue_flag_note_is_last_section(label, prompt):
    assert prompt.endswith(ISSUE_FLAG_PROMPT_NOTE)


def test_note_contains_no_format_braces():
    """The prompts are run through str.format(); a stray brace would raise."""
    assert list(string.Formatter().parse(ISSUE_FLAG_PROMPT_NOTE)) == [
        (ISSUE_FLAG_PROMPT_NOTE, None, None, None)
    ]


def test_note_tells_agent_to_omit_field_when_fine():
    assert "Leave the field out" in ISSUE_FLAG_PROMPT_NOTE
