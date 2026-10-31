# Show-Structure Conventions — vox_romana Reference

**Version:** 1.0 · **Last updated:** 2026-10-31 · **Status:** research reference

This document is the companion to the *Discourse Principles* reference (the
prompt-engineering notes on collaborative-vs-adversarial tone, improv rules,
classical rhetoric, and broadcast techniques). Where that companion covers
**how hosts speak**, this document covers **how a show is shaped** — the
structural conventions vox_romana's generators draw on, organized by medium
and then mapped to concrete templates per shipped show type.

Shipped show types: `debates`, `news`, `podcast`, `audiobooks`, `lecture`.
(`radio_drama` is a planned show type — see the phase-5 backlog item.)

---

## 1. Why structure matters for generated audio

vox_romana generates audio-only content, which constrains structure more than
video does:

- **No visual anchors.** Listeners can't see who is talking or where a segment
  begins — so structure must be *audible*: name cues, transition phrases,
  music/sting placement, and explicit sign-posting.
- **Linear and ephemeral.** There is no re-reading; a listener who gets lost
  cannot skip back easily. Structural clarity is a retention feature, not a
  nicety.
- **Rhythm is the container.** Structure is experienced as pacing (breath
  units, energy arcs, segment length), not as a table of contents.

**Prime directive for every template below:** the structure must be legible
to a listener who is doing something else with their eyes.

---

## 2. Cross-medium conventions

### 2.1 Theater — Freytag's pyramid and scene craft

| Stage | Function | Audio equivalent |
|-------|----------|------------------|
| **Exposition** | Establish world, characters, stakes | Cold/warm open: who's talking, what's tonight's topic, why it matters |
| **Inciting incident** | The event that sets motion | The hook — a specific question, tension, or "moment of mystery" (Ira Glass) |
| **Rising action** | Complications deepen stakes | Deep-dive beats; hosts add counterpoints, examples, anecdotes |
| **Climax** | Point of highest tension | The "but/and" turn — where the conversation finds its most interesting friction or insight |
| **Falling action** | Consequences play out | Takeaways, resolution of loose threads |
| **Denouement** | Return to stasis | Outro: sign-off, next-episode tease |

**Prompt principle:** every generated segment should know *where it sits* on
this arc. A segment that doesn't know whether it's rising, peaking, or
resolving will drift flat or (worse) resolve too early — see §5.2 on false
sign-off.

### 2.2 Hollywood — three-act structure and beat sheets

- **Act I (setup, ~25%)**: establish the normal world, introduce the
  protagonist and the central question, then break the normal world with the
  inciting incident.
- **Act II (confrontation, ~50%)**: escalating obstacles; the midpoint raises
  the stakes; the "all is lost" beat near the end of the act.
- **Act III (resolution, ~25%)**: the climax, then the denouement.

**Beat-sheet notes (Save the Cat family)** useful for generation:

- **Opening image** → the show's opening line should set tone before content.
- **Theme stated** → name the show and topic early (the "promise" of the
  episode).
- **Debate / the "fun and games" section** → the section where the premise
  delivers on its promise; in podcasts this is the deep-dive or banter.
- **Midpoint** → a false victory or false defeat that reframes the topic.
- **Dark night of the soul** → a low-energy reflective beat before the climax.

**Prompt principle:** acts are *not* segments. A 3-segment podcast still maps
to three acts (setup → confrontation → resolution); a 10-segment podcast maps
multiple segments into Act II. Structure scales by *proportion*, not by count.

### 2.3 Radio — clock structure and the "clock"

Radio formats are timed to a **clock** — a circular minute-by-minute map of a
program hour. Conventions that transfer:

- **Hard opens vs. soft opens.** A *hard open* begins mid-topic (a clip, a
  question, a line of dialogue) and is followed by the formal intro; a *soft
  open* is the traditional "welcome to the show." (See §5.1 on cold opens.)
- **Forward promotion / teases.** The host sells what's coming ("later in the
  hour…") to hold the audience through a break. In non-ad-supported shows this
  becomes "we'll get to that" or an explicit teaser of the deep-dive.
- **Sting / bumper placement.** Short audio identifiers mark transitions. In
  vox_romana, `["intro music"]` and `["outro music"]` nonverbal cues are the
  generated equivalents; stings should bracket, not interrupt, speech.
- **Daypart conventions.** Morning = tight and energetic; afternoon =
  longer-form and relaxed; evening = reflective. This maps to the `tone` and
  `duration_hint` fields rather than to a literal clock.
- **Back-sell and re-identify.** After any break, radio re-names the station
  and the show. In a generated multi-segment show, re-naming a host or the
  show after a topic shift serves the same recovery function.

### 2.4 Podcasting — native structural vocabulary

- **Cold open vs. warm open.** The cold open is a preview clip or "we're about
  to talk about X"; the warm open starts at zero and eases in. Both are
  legitimate — what matters is that the show and topic get named before
  substance lands.
- **Segment / chapter breaks.** Long episodes use explicit segments; each
  segment has its own mini-arc (hook → body → button). The "button" is the
  line that closes the segment and pivots to the next.
- **The tease.** A strong cold open ends on a promise the body must keep.
- **Midroll / transition conventions.** Even without ads, the "we'll come back
  to that" convention keeps threads alive across segments.
- **Outro conventions.** Recurring sign-off language, a next-episode tease,
  and a call to action are the genre's ritual closing — and **only** belong
  at the end (see §5.2).

---

## 3. Template: show arc (the master map)

Every generated show, regardless of type, should satisfy this arc. Specific
show types in §4 are refinements of it.

```
HOOK ──► CONTEXT ──► BODY ──► RESOLUTION ──► CLOSE
  │         │           │          │            │
  │         │           │          │            └─ sign-off, tease, outro music
  │         │           │          └─ takeaways, loose threads tied
  │         │           └─ the substance: segments, turns, chapters
  │         └─ name show + topic, introduce speakers, set stakes
  └─ open strong: mystery, question, or promise (never a cold plunge)
```

**Proportions (target, not rule):** hook+context ~15%, body ~60%,
resolution+close ~25%. These proportions already underpin the conductor
word-budget logic (`WRAP_UP_THRESHOLD`, `_rebalanced_target`).

---

## 4. Structural templates by show type

### 4.1 Debate (`debates`)

**Genre arc:** adversarial, turn-based. Structure is a *scored exchange*, not a
collaborative exploration.

| Beat | Content | Structural note |
|------|---------|-----------------|
| Open | Moderator names the topic and the motion | Establish what's actually being contested |
| Stakes | Why this debate matters | Ground rules + importance |
| Opening statements | Each side frames their position | Fixed order, equal time |
| Rebuttal | Cross-examination / counter-points | Turn-by-turn; the tension engine |
| Rebuttal 2 / deepening | Narrower, sharper exchange | Heightening, not escalation |
| Closing statements | Each side's final synthesis | No new evidence |
| Adjudication | Moderator/verdict | Optional in some formats |

**Conventions to enforce:** no new claims in closing; moderator owns the
transitions; rebuttals must reference the *previous* point (listen to build,
not to rebut — see discourse principles).

### 4.2 News (`news`)

**Genre arc:** broadcast rundown. Structure is *discrete stories stacked* in a
deliberate order, each with a mini-arc.

| Beat | Content | Structural note |
|------|---------|-----------------|
| Open | Anchor greets, names the show | Warm open, then headline tease |
| Headline tease | 2-3 top stories previewed | Sells the rundown |
| Story 1 | Lead story: hook → body → context | Hardest-hitting first |
| Story 2..N | Descending importance | Each is self-contained |
| Weather/soft | Human-interest closer | The "kick with a smile" |
| Close | Recap + sign-off | Ritualized, brief |

**Conventions to enforce:** stories are independent — a listener tuning in
mid-show should still understand story N; each story has its own
hook/body/button; the rundown order encodes importance.

### 4.3 Podcast (`podcast`)

**Genre arc:** multi-host collaborative conversation, segment-level.

| Segment type | Role in the arc | Overlay (`SEGMENT_TYPE_OVERLAYS`) |
|--------------|-----------------|-----------------------------------|
| `intro` | Hook + context: name show/topic, host self-intros, tease | `INTRO_OVERLAY` |
| `banter` | Warm-up, personality, light riffs | `BANTER_OVERLAY` |
| `deep_dive` | The body: substance, examples, discovery | `DEEP_DIVE_OVERLAY` |
| `drama` | Climax: high-stakes mutual investment | `DRAMA_OVERLAY` |
| `outro` | Resolution + close: takeaways, sign-off, outro music | `OUTRO_OVERLAY` |

**Conventions to enforce (this doc's direct output):**

1. **Warm, not cold, open** — the show and topic must be named before
   substantive discussion; every host self-introduces in their own voice in
   the intro. A cold plunge into a story or argument is a bug (§5.1).
2. **Sign-off is outro-only** — farewell language ("thanks for listening",
   "that's our show", "until next time") belongs *only* in the final OUTRO
   segment. Mid-show sign-off is a bug (§5.2).
3. **Segment transitions are audible** — a segment's final line should
   "button" the segment and pivot to the next; the listener should never feel
   a hard cut between topics.

### 4.4 Audiobook (`audiobooks`)

**Genre arc:** single-narrator, chapter-by-chapter narrative. Structure is
*narrative* (Freytag) rather than segmental.

| Element | Convention |
|---------|------------|
| Chapter open | Re-orient the listener: where are we, who's here |
| Chapter body | Scene beats: goal → obstacle → reaction |
| Chapter close | A hook or cliffhanger that pulls into the next chapter |
| Book arc | Rising action → climax → denouement across chapters |

**Conventions to enforce:** each chapter must be able to stand alone *and*
advance the book arc; chapter breaks are natural pauses (TTS chunk
boundaries), so end chapters on a beat that survives a pause.

### 4.5 Lecture (`lecture`)

**Genre arc:** single-speaker pedagogy. Structure is *didactic* (signposted
learning), not narrative.

| Element | Convention |
|---------|------------|
| Open | State the thesis / learning objective up front |
| Roadmap | Preview the sections ("we'll cover A, B, C") |
| Section | Claim → explanation → example → restatement |
| Signposts | "Now", "moving on", "to recap" — audible scaffolding |
| Recap | Summarize before moving on |
| Close | Synthesize and restate the takeaway |

**Conventions to enforce:** no stage directions (lecture already forbids
them); the roadmap must match the delivered sections (a promised section that
never appears is a listener-trust failure); signposting is a feature, not
filler.

---

## 5. Known failure modes and their structural fixes

### 5.1 Abrupt / cold open (0→100 mph)

**Symptom:** the show dives into the topic before naming the show, the topic,
or the speakers.

**Structural cause:** skipping the hook+context beats and starting in the body.

**Fix (applied):** `INTRO_OVERLAY` now requires the show and topic to be named
in the first utterance(s) and every host to self-introduce before substance.
The system prompt forbids cold opens outright.

### 5.2 False sign-off mid-show

**Symptom:** a host says "thanks for listening" or "that's our show" with
significant content still remaining.

**Structural cause:** resolution/close language leaking into body beats —
i.e., the model resolving the arc too early.

**Fix (applied):** the system prompt restricts sign-off language to the OUTRO
segment; `SEGMENT_PROMPT` carries an explicit `SEGMENT TYPE` line so the model
knows where it sits; and `PodcastConductor._check_false_signoff()` flags
mid-show sign-off language on the warnings channel (non-blocking).

### 5.3 Flat or meandering body

**Symptom:** no clear peak; the conversation drifts without rising tension.

**Structural cause:** no midpoint/climax beat — Act II never escalates.

**Fix:** ensure the segment plan includes a `drama`-type segment (podcast) or
an explicit rebuttal-deepening round (debate), and that director notes
(`_director_check`) can nudge the energy when the arc flatlines.

### 5.4 Hard cuts between topics

**Symptom:** the conversation jumps topics with no pivot.

**Structural cause:** missing segment "buttons" — the closing line of one
segment doesn't hand off to the next.

**Fix:** the lead host should close each segment with a pivot line ("but the
part I keep coming back to…", "which brings us to…") that names the next
thread.

---

## 6. How to use this reference

1. **When editing prompts**, check the template in §4 for the show type you're
   touching; the structural rules there are the contract the generator must
   satisfy.
2. **When adding a new show type** (e.g. `radio_drama`), pick the closest
   template (§4) and adapt its beats, then document the deltas here.
3. **When debugging structure**, look at §5 first — the four failure modes
   map directly to prompt gaps that have already bitten generated shows.
4. **When tuning length**, keep the §3 proportions in mind: structure scales
   by proportion, and the conductor's word-budget machinery already assumes
   that hook+context ~15%, body ~60%, resolution+close ~25%.

---

## 7. Sources and further reading

- **Theater:** Gustav Freytag, *Technik des Dramas* (1863) — five-stage
  dramatic structure.
- **Hollywood:** Syd Field, *Screenplay* (three-act); Blake Snyder, *Save the
  Cat!* (beat sheet).
- **Radio:** Robert McLeish, *Radio Production* (clock structure, daypart
  conventions).
- **Podcasting:** Ira Glass on storytelling (the "moment of mystery"); NPR
  training materials on cold opens and teases.
- **vox_romana internal:** the *Discourse Principles* reference (improv rules,
  classical rhetoric, broadcast rhetoric) is this document's companion — it
  covers *how* hosts speak; this covers *how* shows are shaped.
