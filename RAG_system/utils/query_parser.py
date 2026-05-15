from typing import Dict, Optional

import spacy


_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe(
            "entity_ruler",
            config={"phrase_matcher_attr": "LOWER"},
            before="ner",
        )
        ruler.add_patterns(
            [
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "senior"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "sr"}, {"IS_PUNCT": True, "OP": "?"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "entry"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "junior"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "jr"}, {"IS_PUNCT": True, "OP": "?"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "mid"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "internship"}]},
                {"label": "JOB_LEVEL", "pattern": [{"LOWER": "intern"}]},

            ]
        )
        _nlp = nlp
    return _nlp


_LEVEL_MAP = {
    "senior": "Senior Level",
    "sr": "Senior Level",
    "entry": "Entry Level",
    "junior": "Entry Level",
    "jr": "Entry Level",
    "mid": "Mid Level",
    "intern": "Internship",
    "internship": "Internship",
}

_NON_COMPANY_ORGS = {
    "software", "engineering", "data", "design", "ai", "ml",
    "product", "marketing", "sales", "research", "development",
    "finance", "hr", "operations", "strategy", "consulting",
    "architecture", "analytics", "infrastructure", "platform",
    "security", "quality", "testing", "devops", "agile",
}


def extract_filters(query: str) -> Dict[str, str]:
    doc = _get_nlp()(query)
    filters = {}

    for ent in doc.ents:
        text_lower = ent.text.lower()

        if ent.label_ == "JOB_LEVEL" and "level" not in filters:
            mapped = _LEVEL_MAP.get(text_lower)
            if mapped:
                filters["level"] = mapped

        elif ent.label_ == "ORG" and "company" not in filters:
            if text_lower not in _NON_COMPANY_ORGS:
                filters["company"] = ent.text

    return filters
