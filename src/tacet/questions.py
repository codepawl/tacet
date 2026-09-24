"""Builders for the three question shapes. Plain dicts work just as well."""


def choice(instructions, options):
    """Pick one option out of a named set: {"name": "description", ...}."""
    return {"type": "choice", "instructions": instructions, "criteria": dict(options)}


def score(instructions, levels):
    """Place the state on an ordered list of levels, lowest first."""
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


def noul(instructions, criteria=None):
    """Yes or no. `criteria` may say what "true" and "false" mean."""
    question = {"type": "noul", "instructions": instructions}
    if criteria is not None:
        question["criteria"] = dict(criteria)
    return question
