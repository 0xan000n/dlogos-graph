"""Cross-episode speaker identity (hosts + recurring guests).

Diarization (``dlogos.asr``) only gives *within-episode* labels
(``SPEAKER_00`` …). This subpackage resolves those labels to canonical
:class:`~dlogos.schema.SpeakerRef`\\s that are stable *across* episodes — the
prerequisite for "what does *[person]* believe about X, and has it changed?"

Two complementary resolvers (spec §7.3 / §7.4a):

- :mod:`dlogos.speakers.identity` — **host-anchored voiceprint gallery**. A
  gallery seeded from the corpus manifest's ``known_hosts`` matches each
  episode's diarization labels to canonical hosts by cosine similarity. The
  voice embedder is *injected* (a :class:`VoiceEmbedder` protocol) so tests use
  a deterministic fake.
- :mod:`dlogos.speakers.guests` — **recurring-guest resolution** from three
  cheap signals (episode metadata, the spoken "my guest today is …" intro, and
  a Wikidata QID match). Recurring high-value guests resolve to a stable id;
  the one-off long tail stays per-episode.

Diarization error here is the corpus's *top correctness risk* (spec §11): a
swapped label produces a confident *misattribution*. Both resolvers are
deliberately conservative — they prefer leaving a label unresolved over a wrong
merge.
"""

from __future__ import annotations

from dlogos.speakers.guests import (
    GuestCandidate,
    GuestResolution,
    GuestResolver,
    WikidataClient,
    WikidataMatch,
    extract_intro_names,
)
from dlogos.speakers.identity import (
    CanonicalSpeaker,
    HostGallery,
    HostSeed,
    SpeakerResolution,
    VoiceEmbedder,
)

__all__ = [
    # identity
    "CanonicalSpeaker",
    "HostGallery",
    "HostSeed",
    "SpeakerResolution",
    "VoiceEmbedder",
    # guests
    "GuestCandidate",
    "GuestResolution",
    "GuestResolver",
    "WikidataClient",
    "WikidataMatch",
    "extract_intro_names",
]
