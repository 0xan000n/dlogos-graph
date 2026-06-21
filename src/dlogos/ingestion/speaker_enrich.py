"""Map raw transcript speaker *labels* to known full names (pure, stdlib).

The text transcripts carry real speaker names, but the form varies by source:

- The Jim Rutt Show renders **bare first names** ("Jim", "Nate") — fine in
  isolation but ambiguous across the corpus, where "Nate" is Nate Soares in one
  episode and Nate Hagens in another (and "Nathan" Lambert elsewhere).
- Some PDFs shout names in **all caps** ("TRISTAN HARRIS").
- A handful of one-off lines are *not speakers at all* ("Researchers",
  "Companies", "Governments") or are archival-clip attributions ("Robert
  Oppenheimer") that match nobody on the episode's roster.

:func:`enrich_speakers` resolves each segment's label against the episode's
**known speakers** (its host + guests, full names) and rewrites the label to the
matching full name — but *only* when the match is unambiguous. The match
strategy, tried in order, is:

1. **exact** (casefold): ``"tristan harris"`` / ``"TRISTAN HARRIS"`` →
   ``"Tristan Harris"``.
2. **first name**: the label equals the first token of **exactly one** known
   name → that name. ``"Jim"`` → ``"Jim Rutt"``; ``"Nate"`` → ``"Nate Soares"``
   when Soares is the only "Nate" on this episode's roster.
3. **surname**: the label equals the last token of **exactly one** known name.
4. **unique substring**: the casefolded label is a substring of **exactly one**
   known name (and of no other) → that name.

If a label matches **none** of the known speakers, or matches **more than one**
(ambiguous), the label is left **unchanged** — the function never guesses. This
keeps the stray non-speaker labels and unresolvable archival names exactly as
they appeared on the page, while collapsing the ambiguous first names onto the
right full identity for the name-driven resolver downstream.

Pure and deterministic: same ``(segments, known_speakers)`` → same output, no
network / filesystem / clock. Stdlib only.
"""

from __future__ import annotations

from dlogos.schema import TranscriptSegment

__all__ = ["enrich_speakers", "resolve_label"]


def _first_token(name: str) -> str:
    parts = name.split()
    return parts[0].casefold() if parts else ""


def _last_token(name: str) -> str:
    parts = name.split()
    return parts[-1].casefold() if parts else ""


def _unique(matches: list[str]) -> str | None:
    """The single element of ``matches`` if there is exactly one, else ``None``.

    De-dups first (the same full name reached via two rules is still one match),
    so ``["Jim Rutt", "Jim Rutt"]`` resolves but ``["Nate Soares", "Nate
    Hagens"]`` stays ambiguous → ``None``.
    """

    distinct = list(dict.fromkeys(matches))
    return distinct[0] if len(distinct) == 1 else None


def resolve_label(label: str, known_speakers: list[str]) -> str:
    """Resolve one speaker ``label`` to a known full name, or return it unchanged.

    Tries, in order: exact (casefold) → first-name → surname → unique substring,
    against ``known_speakers``. Returns the matching full name only when exactly
    one known speaker matches at the first rule that hits; otherwise (no match or
    ambiguous) returns ``label`` verbatim. Pure and deterministic.
    """

    raw = label.strip()
    if not raw:
        return label
    key = raw.casefold()

    known = [k for k in known_speakers if k and k.strip()]
    if not known:
        return label

    # 1) exact (casefold). A label that already equals a known name wins outright
    #    — covers "TRISTAN HARRIS" → "Tristan Harris" and idempotent re-runs.
    exact = [k for k in known if k.casefold() == key]
    if exact:
        # If the page already names them fully, prefer the canonical spelling
        # from the roster; ambiguity here would mean a duplicate roster entry.
        resolved = _unique(exact)
        if resolved is not None:
            return resolved

    # 2) first name: label == first token of exactly one known name.
    by_first = [k for k in known if _first_token(k) == key]
    resolved = _unique(by_first)
    if resolved is not None:
        return resolved
    if by_first:
        # Ambiguous first name ("Nate" with two Nates) — never guess.
        return label

    # 3) surname: label == last token of exactly one known name.
    by_last = [k for k in known if _last_token(k) == key]
    resolved = _unique(by_last)
    if resolved is not None:
        return resolved
    if by_last:
        return label

    # 4) unique substring: casefolded label appears in exactly one known name.
    by_sub = [k for k in known if key in k.casefold()]
    resolved = _unique(by_sub)
    if resolved is not None:
        return resolved

    # No match, or ambiguous — conservative: leave the label as it was.
    return label


def enrich_speakers(
    segments: list[TranscriptSegment], known_speakers: list[str]
) -> list[TranscriptSegment]:
    """Rewrite each segment's speaker label to the best-matching known full name.

    For every segment, :func:`resolve_label` maps its ``speaker`` against
    ``known_speakers`` (the episode's host + guests). A unique match becomes the
    full name; no match or an ambiguous one leaves the label unchanged (the
    function never guesses). All other segment fields (``text``, ``t_start``,
    ``t_end``) are preserved.

    Returns a **new** list of new :class:`~dlogos.schema.TranscriptSegment`
    objects (inputs are not mutated). With an empty ``known_speakers`` every
    label is returned verbatim. Pure and deterministic.
    """

    if not known_speakers:
        return list(segments)

    out: list[TranscriptSegment] = []
    for seg in segments:
        resolved = resolve_label(seg.speaker, known_speakers)
        if resolved == seg.speaker:
            out.append(seg)
        else:
            out.append(seg.model_copy(update={"speaker": resolved}))
    return out
