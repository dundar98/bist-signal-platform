"""Utilities module for the BIST100 trading system."""

import unicodedata

from .logging import (
    setup_logging,
    get_logger,
    TradeLogger,
)


def normalize_ticker(text: str) -> str:
    """Normalize a ticker/query string to ASCII-uppercase, handling Turkish characters.

    Turkish-specific mappings:
        'ı' (U+0131) → 'I'
        'i'  → 'I'
        'İ' (U+0130) → 'I'
        'ş' → 'S', 'Ş' → 'S'
        'ğ' → 'G', 'Ğ' → 'G'
        'ü' → 'U', 'Ü' → 'U'
        'ö' → 'O', 'Ö' → 'O'
        'ç' → 'C', 'Ç' → 'C'

    All other accented characters are stripped to their ASCII base via NFKD
    normalization.  The result is a plain ASCII-uppercase string suitable for
    ticker lookups and database comparisons.
    """
    # Manual Turkish-specific mappings that unicodedata misses or handles
    # inconsistently across Python versions / locales.
    turkish_map = {
        "\u0130": "I",  # İ (dotted capital I)
        "\u0131": "I",  # ı (dotless lowercase i)
        "\u015e": "S",  # Ş
        "\u015f": "S",  # ş
        "\u011e": "G",  # Ğ
        "\u011f": "G",  # ğ
        "\u00dc": "U",  # Ü
        "\u00fc": "U",  # ü
        "\u00d6": "O",  # Ö
        "\u00f6": "O",  # ö
        "\u00c7": "C",  # Ç
        "\u00e7": "C",  # ç
    }
    text = text.strip()
    # Strip known exchange suffixes (.IS, .E, .TI)
    for suffix in (".IS", ".E", ".TI"):
        if text.upper().endswith(suffix):
            text = text[: -len(suffix)]
            break

    result = []
    for ch in text:
        if ch in turkish_map:
            result.append(turkish_map[ch])
        else:
            # NFKD normalization decomposes accented chars (à→a, é→e, etc.)
            decomposed = unicodedata.normalize("NFKD", ch)
            # Take the first (base) character, drop combining marks
            base = decomposed[0] if decomposed else ch
            # Only keep ASCII letters, digits, and common symbols
            if base.isascii() and (base.isalnum() or base in ".-_"):
                result.append(base.upper())
            elif base.isascii():
                result.append(base.upper())
            else:
                # Fallback: keep as-is for non-Latin (shouldn't happen for BIST)
                result.append(base.upper())
    return "".join(result)


__all__ = [
    "setup_logging",
    "get_logger",
    "TradeLogger",
    "normalize_ticker",
]
