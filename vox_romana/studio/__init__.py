"""vox_romana.studio — one-sentence studio instantiation.

Shot 5.2-a ships the staged inference prompts + parsers; 5.2-b ships the
orchestrator; 5.2-c ships bundle materialization + zip export.  CLI wiring
lands in 5.2-d.
"""

from .bundle import materialize_bundle, zip_bundle
from .instantiator import StudioInstantiator, StudioInstantiation
from .prompts import (
    AUDIENCE_SYSTEM_PROMPT,
    BIBLE_SYSTEM_PROMPT,
    META_SYSTEM_PROMPT,
    StudioMeta,
    build_audience_user_prompt,
    build_bible_user_prompt,
    build_meta_user_prompt,
    parse_audience,
    parse_bible,
    parse_meta,
)

__all__ = [
    "AUDIENCE_SYSTEM_PROMPT",
    "BIBLE_SYSTEM_PROMPT",
    "META_SYSTEM_PROMPT",
    "StudioInstantiation",
    "StudioInstantiator",
    "StudioMeta",
    "build_audience_user_prompt",
    "build_bible_user_prompt",
    "build_meta_user_prompt",
    "materialize_bundle",
    "parse_audience",
    "parse_bible",
    "parse_meta",
    "zip_bundle",
]
