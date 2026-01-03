from __future__ import annotations

import unicodedata


def _is_letter_or_digit(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat and cat[0] in {"L", "N"}


def _is_symbol(ch: str, count_space: bool = False) -> bool:
    if count_space:
        return not _is_letter_or_digit(ch)
    return not _is_letter_or_digit(ch) and not ch.isspace()


def dirty_reason(
    text: str,
    token_length: int,
    *,
    unique_token_ratio: float,
    symbol_ratio: float,
    max_consecutive_symbols: int,
    max_repeated_char_run: int,
    source: str | None = None,
) -> str | None:
    pd_source = bool(source and "pd" in source.lower())

    if token_length == 0:
        return "zero_length"

    tokens = [t for t in text.split() if t]
    if not tokens:
        return "no_tokens"

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < unique_token_ratio:
        return f"low_unique_ratio:{unique_ratio:.3f}"

    if pd_source:
        symbol_count = sum(1 for c in text if _is_symbol(c, count_space=True))
    else:
        symbol_count = sum(1 for c in text if _is_symbol(c))
    sym_ratio = symbol_count / max(len(text), 1)
    if sym_ratio > symbol_ratio:
        return f"high_symbol_ratio:{sym_ratio:.3f}"

    max_symbol_run = 0
    cur_symbol_run = 0
    max_repeat_run = 0
    prev_char = ""
    cur_repeat = 0

    for ch in text:
        if pd_source:
            if _is_symbol(ch, count_space=False):
                cur_symbol_run += 1
                if cur_symbol_run > max_symbol_run:
                    max_symbol_run = cur_symbol_run
            else:
                cur_symbol_run = 0
        else:
            cur_symbol_run = 0

        if ch == prev_char and not ch.isspace():
            cur_repeat += 1
            if cur_repeat > max_repeat_run:
                max_repeat_run = cur_repeat
        else:
            cur_repeat = 1
            prev_char = ch

        if pd_source and max_symbol_run > max_consecutive_symbols:
            return f"long_symbol_run:{max_symbol_run}"
        if max_repeat_run > max_repeated_char_run:
            return f"long_repeat_run:{max_repeat_run}"

    return None
