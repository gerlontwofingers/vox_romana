"""Podcast system prompt templates.

Podcasts are REACTIVE multi-host conversations. Unlike debates (adversarial,
turn-by-turn), podcasts generate an entire segment in one LLM call — producing
multiple speakers' lines in a single response. The key prompt innovation is
the SEGMENT_PROMPT, which instructs the LLM to return a JSON array of
utterances from multiple speakers.

Unlike debate prompts (one speaker per call), podcast prompts ask the LLM to
orchestrate ALL hosts within a segment simultaneously. This means the system
prompt sets up the FULL cast, and the segment prompt says "now have them
discuss X."

These templates use str.format() placeholders. The conductor fills them in
at runtime with data from the PodcastConfig and host profiles.
"""

# ═══════════════════════════════════════════════════════════════════════════
# PODCAST HOST SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# This is the "cast sheet" — injected as the system prompt for the
# segment-generating agent. It describes ALL hosts at once, because the
# LLM needs to voice everyone in a single response.

PODCAST_HOST_SYSTEM_PROMPT = """\
You are writing dialogue for a podcast called "{show_name}" about: {topic}

TONE: {tone}

THE HOSTS (you write lines for ALL of them):
{host_descriptions}

RULES:
- Write natural, conversational dialogue suitable for audio recording.
- Each host has their own personality, speech style, and expertise —
  make them sound DISTINCT from each other.
- Hosts should RESPOND to each other. This is a conversation, not parallel
  monologues. They react, build on each other's points, ask follow-up
  questions, and sometimes offer a different angle — always from curiosity,
  not from the desire to win.
- Use hosts' NAMES naturally in conversation — this is audio-only, so the
  listener needs periodic name cues to track who's talking. Work names
  into responses like real people do: "Good point, Maven..." or "Something
  Chill Carlos said earlier..."
- Respect each host's KNOWLEDGE DOMAINS — they speak authoritatively on
  their domains and defer or ask questions outside them.
- Respect FORBIDDEN KNOWLEDGE — hosts NEVER claim expertise in their
  forbidden topics. They can react ("I don't know much about cricket, but...")
  but must not lecture.
- Vary energy levels. Not every line needs to be high-energy. Mix up:
  reflective moments, excited tangents, quiet agreement, playful teasing.

CONVERSATION PHILOSOPHY:
- This is NOT a debate. Hosts are exploring a topic together — "winning"
  an argument is a failure mode, not a goal.
- BUILD FIRST: when another host makes a point, find what is interesting
  and true in it before departing from it. Even disagreement should expand
  the idea, not shut it down.
- CURIOSITY AS DEFAULT STANCE: when uncertain, ask. "What do you mean
  by...?" is more interesting than a prepared counter-argument. Genuine
  not-knowing is a feature, not a weakness.
- YES, AND: find the truth in what was just said, then add to it. "Yes,
  and actually that reminds me of..." beats "Actually, no, I think..."
- PROTECT YOUR PARTNER: ask questions that let other hosts deepen their
  point, not corner them. Leave room and exits.
- LISTEN TO BUILD, NOT TO REBUT: reference something specific the previous
  speaker just said before giving your own take. React to what was
  actually said, not to the argument you were already preparing.

ANTI-REPETITION:
- NEVER repeat a phrase, talking point, or argument already used.
- VARY sentence openers. No two consecutive lines should start the same way.
- Once a point is made, ADVANCE to a new angle: a specific example, a
  personal anecdote, a counterpoint, a question, a tangent.
- Draw from each host's TANGENT TOPICS when the main thread is covered.

FACTUAL ACCURACY — CRITICAL:
  Today's date is {current_date}.
- DO NOT invent names of real people, companies, studies, statistics, court
  cases, government programmes, URLs, or domain names. If you are not
  certain something is real, do NOT state it as fact — say "I think" or
  ask the other host.
- DO NOT fabricate biographical details (birth dates, death dates, cause of
  death, job titles) about any named person.
- DO NOT invent revenue figures, user counts, market sizes, or other
  statistics. If the reference material does not supply a number, say you
  don't have the exact figure.
- DO NOT backdate events or misstate when someone died. Use {current_date}
  as your anchor — anyone described as a recent death must have died on or
  before this date.
- WHEN IN DOUBT: have the host hedge explicitly ("I don't know the exact
  figure but...") rather than guessing a number that sounds plausible.

OUTPUT FORMAT:
Respond with ONLY a JSON array of utterance objects (no other text):
[
  {{
    "speaker_name": "The Maven",
    "text": "spoken words here",
    "tone": "neutral|calm|warm|excited|skeptical|passionate|playful|contemplative|urgent|dramatic",
    "volume": "soft|moderate|loud|rising|falling",
    "pace": "slow|medium|fast|accelerating|decelerating — default to slow or medium; reserve fast for genuine excitement or urgency",
    "nonverbal": []
  }},
  ...
]

NONVERBAL FIELD — these are AUDIBLE cues only (no visual gestures). Use
sparingly to add color: e.g. ["laughs"], ["chuckles"], ["sighs"],
["deep breath"], ["groans"], ["clears throat"], ["whispers"],
["intro music"], ["outro music"]. Most utterances should leave it empty
([]). Roughly 1 in 4-6 utterances may carry one cue when it genuinely
shapes the moment. Never use visual cues like "gestures", "nods",
"leans forward" — the listener cannot see them.

Each object is one speaker turn. Produce 3-15 turns per segment, depending
on the segment type and target length. Every host should speak at least once
in multi-host segments.
"""


# ═══════════════════════════════════════════════════════════════════════════
# HOST DESCRIPTION BLOCK
# ═══════════════════════════════════════════════════════════════════════════
# Template for describing one host within the system prompt. The conductor
# fills one of these per host and joins them into {host_descriptions}.

HOST_DESCRIPTION_TEMPLATE = """\
  {name} ({role}):
    Style: {speech_style}
    Voice: {voice_fingerprint}
    Expertise: {knowledge_domains}
    Off-limits: {forbidden_knowledge}
    Interruption tendency: {interruption_tendency}
    Personality triggers: {behavioral_triggers}
    Tangent topics: {tangent_topics}
"""


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENT PROMPT — injected per segment call
# ═══════════════════════════════════════════════════════════════════════════
# This is the user-message sent for each segment. It tells the LLM what
# kind of segment to produce (intro, banter, deep_dive, etc.) and provides
# context from previous segments.

SEGMENT_PROMPT = """\
Continue the conversation.

TOPIC: {segment_topic}
LEAD HOST: {lead_host}

{length_contract}

{type_overlay}

CONTEXT (what's happened so far):
{context_summary}

{director_note}

{runtime_status}

Remember: respond with ONLY a JSON array of utterance objects. Each host \
should contribute naturally. The lead host drives the segment but others \
react, interject, and build on points.
"""


# ═══════════════════════════════════════════════════════════════════════════
# SEGMENT TYPE OVERLAYS
# ═══════════════════════════════════════════════════════════════════════════
# Short type-specific instructions appended to the segment prompt. These
# set the tone and structural expectations for each segment type.

INTRO_OVERLAY = """\
INTRO STRUCTURE:
- Open with energy and personality — hook the listener immediately.
- Name the show and tonight's topic.
- Each host should introduce themselves naturally (not robotically):
  a quip, a tease, a personal connection to the topic.
- Build anticipation for what's coming. Tease a specific angle or hot take.
- Keep it crisp — intros should run 3-6 speaker turns total.
- On the VERY FIRST utterance object, include `"nonverbal": ["intro music"]`
  to signal that intro music should play at the top of the show.
"""

BANTER_OVERLAY = """\
BANTER GUIDELINES:
- Light, personality-driven conversation. This is the "hang out" segment.
- Hosts riff on a topic, share anecdotes, tease each other.
- Let the conversation meander naturally — tangents are welcome.
- Energy should feel relaxed but engaged. Think friends talking, not panelists.
- Disagreements here are playful, not heated.
"""

DEEP_DIVE_OVERLAY = """\
DEEP DIVE GUIDELINES:
- This is the meat of the episode. Go deep on the topic.
- The lead host drives, but others contribute expertise and reactions.
- Use specific examples, data points, stories. Avoid vague generalizations.
- Build arguments step by step. It's OK to get nerdy or technical.
- If covering multiple sub-topics, transition naturally between them.
- This segment should feel substantial — the listener should learn something.
- Hosts are discovering and making sense of this together, not reciting
  prepared positions. Thinking out loud is encouraged.
"""

DRAMA_OVERLAY = """\
DRAMA GUIDELINES:
- High stakes. Both hosts care DEEPLY about this topic, possibly from
  different angles. The energy comes from that investment, not from
  opposition.
- Disagreements here are about what MATTERS, not about who is right.
  The question is "what should we do about this?" not "who is correct?"
- Interruptions are natural when something lands hard — respect each
  host's interruption_tendency.
- Let intensity build through mutual investment: "I agree with you AND
  it's even more serious than that..." is more dramatic than "No, you're
  wrong."
- End with the feeling that something important was said — not a winner
  declared.
"""

OUTRO_OVERLAY = """\
OUTRO GUIDELINES:
- Wind the episode down. Energy shift from the previous segments.
- Each host gives a final take or takeaway (1-2 sentences each).
- Thank the listener naturally — don't be robotic about it.
- Tease next episode or leave the listener with something to think about.
- Sign off with the show name. Keep it warm and genuine.
"""

# Map segment types to their overlay text for easy lookup.
SEGMENT_TYPE_OVERLAYS = {
    "intro": INTRO_OVERLAY,
    "banter": BANTER_OVERLAY,
    "deep_dive": DEEP_DIVE_OVERLAY,
    "drama": DRAMA_OVERLAY,
    "outro": OUTRO_OVERLAY,
}


# ═══════════════════════════════════════════════════════════════════════════
# CONTEXT SUMMARY PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# Used to compress older segment summaries into a concise context block.
# Same pattern as audiobooks and news — keeps the sliding window tight.

CONTEXT_SUMMARY_PROMPT = """\
Summarize the podcast conversation so far in 2-3 concise sentences.
Focus on: key topics discussed, notable moments or disagreements, the
current energy level, and where the conversation stands. This summary
will be used as continuity context for the next segment.

CONVERSATION SO FAR:
{transcript}
"""


# ═══════════════════════════════════════════════════════════════════════════
# DIRECTOR CHECK PROMPT
# ═══════════════════════════════════════════════════════════════════════════
# The director reviews progress and provides mid-episode pacing notes.

DIRECTOR_CHECK_PROMPT = """\
You are the Director reviewing a podcast episode at segment {segment_number} \
of {total_segments}.

EPISODE ARC:
{arc}

CURRENT MOOD TARGET:
{current_mood}

CLOSING STRATEGY:
{closing_strategy}

RECENT CONTENT (last segment summary):
{recent_content}

Based on the current state, provide a brief stage direction as JSON:
{{
  "mood_note": "1-2 sentence instruction for the current energy/mood",
  "nudge": "optional specific direction for the next segment, or null",
  "wrap_up": false
}}

Set "wrap_up" to true ONLY when it's time to begin the closing segment(s).
"""
