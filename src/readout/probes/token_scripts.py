"""First-codepoint script bucketing for vocabulary tokens.

This is the exact tagger used to build the eval corpora and the script-level
NLL stratification in the temporal-patch experiments (historically duplicated
in the corpus builder and the archived intervention module). The bucketing is
a fixed convention baked into published metrics — codepoint ranges must not
be "improved" (e.g. swapped for unicodedata) without re-deriving those
metrics.
"""

from __future__ import annotations


def script_of(token_text: str) -> str:
    """Coarse script of a token by its first non-space codepoint."""
    if not token_text:
        return "empty"
    s = token_text.lstrip(" ")
    if not s:
        return "space"
    cp = ord(s[0])
    if cp < 128:
        return "ASCII"
    if 0x0400 <= cp <= 0x04FF:
        return "Cyrillic"
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"
    if 0x0600 <= cp <= 0x06FF:
        return "Arabic"
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "Bengali"
    if 0x4E00 <= cp <= 0x9FFF:
        return "CJK"
    if 0x3040 <= cp <= 0x30FF:
        return "Japanese"
    if 0xAC00 <= cp <= 0xD7AF:
        return "Korean"
    return "other_unicode"
