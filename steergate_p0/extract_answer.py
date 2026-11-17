"""Strict \\boxed{} answer extraction and GSM8K grading."""
from __future__ import annotations
import re

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def last_boxed(text: str):
    """Content of the last \\boxed{...} (brace-matched). None if absent or unbalanced (e.g. truncated)."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        c = text[k]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def parse_number(s):
    """Last number in a boxed string. Strips commas, $, %, and LaTeX command names (keeps their arguments)."""
    if s is None:
        return None
    s = s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    nums = _NUM.findall(s)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def gold_from_gsm8k(answer_field: str) -> float:
    return float(answer_field.split("####")[-1].strip().replace(",", ""))


def grade(text: str, gold: float, tol: float = 1e-6) -> dict:
    box = last_boxed(text)
    pred = parse_number(box)
    return {"has_box": box is not None, "pred": pred,
            "correct": bool(pred is not None and abs(pred - gold) <= tol * max(1.0, abs(gold)))}
