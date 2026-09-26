"""Conservative extraction: a cheap relevance filter, then what the text says happened.

Default method is `rules_v1` — keyword rules over the headline (and excerpt when stored). It is
a fallback, not an interpretation: ambiguous results go to manual review, and the method is
always recorded. A language-model adapter can be plugged in (LLMExtractor); its answer is
validated by a strict schema that has no coordinates at all, and invalid answers go to review.
Article text is untrusted data: nothing in it is executed or obeyed."""

from __future__ import annotations

import re
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

ReportType = Literal["occurred", "forecast", "warning", "negation", "historical", "unclear"]

FLOOD = re.compile(
    r"\b(flood\w*|inundat\w*|submerg\w*|deluge|наводнен\w*|паводк\w*|паводок|подтоп\w*|"
    r"затоп\w*|половод\w*|разлив\w*\s+рек\w*|ушл\w*\s+под\s+воду)", re.IGNORECASE)
METAPHOR = re.compile(r"\b(flood of (refugees|migrants|orders|calls|emails|money)|"
                      r"flooded with (calls|messages|requests|orders)|floodlight\w*|"
                      r"flood\s*gate\w* (open|of))", re.IGNORECASE)
NEGATION = re.compile(
    r"(не подтверд\w*|не зафиксир\w*|не зарегистрир\w*|не выявлен\w*|нет подтоплен\w*|"
    r"не затоплен\w*|подтоплени\w* нет|опроверг\w*|фейк\w*|вода ушла|вода отступила|"
    r"no (reports of )?flood\w*|not flooded|den(y|ies|ied)|debunk\w*|fake|hoax|"
    r"false (report|claim)s?|water(s)? (has |have )?receded)", re.IGNORECASE)
FORECAST = re.compile(
    r"(прогноз\w*|ожида\w*|возмож\w*|может (быть )?подтоп\w*|угроз\w*|риск\w*|"
    r"forecast\w*|expected to|could (flood|see)|may flood|likely to|risk of|threat of|"
    r"predict\w*)", re.IGNORECASE)
WARNING = re.compile(
    r"(предупрежд\w*|штормов\w*|оповещ\w*|красн\w+ уров\w*|warning\w*|alert\w*|"
    r"watch in effect|advisory|evacuation order\w*|brace\w* for)", re.IGNORECASE)
HISTORICAL = re.compile(
    r"(годовщин\w*|лет назад|год назад|вспомина\w*|истори\w* паводк\w*|anniversary|"
    r"years ago|years after|remember\w*|look(ing)? back|in (19\d\d|200\d|201\d))", re.IGNORECASE)
OCCURRED = re.compile(
    # verbs and short participles only: the noun «подтопление» alone says nothing happened
    r"(затопил\w*|затоплен[аоы]?\b|подтопил\w*|подтоплен[аоы]?\b|эвакуир\w*|ушл\w* под воду|"
    r"вышл\w* из берегов|flooded|floods? (hit|hits|swamp\w*|kill\w*|ravage\w*|batter\w*|"
    r"inundate\w*|displace\w*|sweep\w*|leave\w*)|submerged|evacuat\w*|stranded|"
    r"under water|killed in flood\w*|flood\w* (death|toll)|dead in flood\w*)", re.IGNORECASE)
# A capitalized place after a locative preposition; kept as a string for the geocoder.
PLACE = re.compile(
    r"(?:\b(?:in|near|across|across the|at|hit|hits|swamp|batter)\s+|"
    r"(?:^|\s)(?:в|во|под|на|у|близ|около)\s+)"
    r"((?:[A-ZА-ЯЁ][\w\-’']+)(?:[ \-](?:[A-ZА-ЯЁ][\w\-’']+)){0,3})")
NOT_PLACES = {"The", "This", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
              "Saturday", "Sunday", "January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"}


class ExtractedTime(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: str | None = None
    end: str | None = None
    precision: Literal["hour", "day", "several_days", "unknown"] = "unknown"


class Extraction(BaseModel):
    """Everything a model or the rules may say. No coordinates, casualties or extent fields:
    the schema itself makes them impossible to return."""
    model_config = ConfigDict(extra="forbid")
    relevant: bool
    report_type: ReportType
    place_mentions: list[str] = Field(default_factory=list, max_length=10)
    event_time: ExtractedTime = Field(default_factory=ExtractedTime)
    summary: str = Field(default="", max_length=400)
    quotes: list[str] = Field(default_factory=list, max_length=5)
    needs_review: bool
    reasons: list[str] = Field(default_factory=list)
    method: str


class LLMExtractor(Protocol):
    """A provider adapter returns raw JSON text for one publication. It gets no tools, no
    network beyond its own API, and cannot change state; the pipeline validates the answer."""
    name: str

    def extract_json(self, title: str, excerpt: str | None) -> str: ...


def validate_llm_answer(raw: str, method: str) -> Extraction | None:
    try:
        data = Extraction.model_validate_json(raw)
    except ValidationError:
        return None
    return data.model_copy(update={"method": method})


def _quote(match: re.Match, text: str, width: int = 60) -> str:
    a, b = max(0, match.start() - width), min(len(text), match.end() + width)
    return text[a:b].strip()


def places(text: str) -> list[str]:
    found: list[str] = []
    for m in PLACE.finditer(text):
        name = m.group(1).strip(" -’'")
        if name.split()[0] in NOT_PLACES or FLOOD.fullmatch(name):
            continue
        if name not in found:
            found.append(name)
    return found


def extract_rules(title: str, excerpt: str | None = None) -> Extraction:
    text = f"{title}. {excerpt}" if excerpt else title
    method = "rules_v1"
    flood = FLOOD.search(text)
    if not flood or (METAPHOR.search(text) and len(FLOOD.findall(text)) <= 1):
        return Extraction(relevant=False, report_type="unclear", needs_review=False,
                          reasons=["нет признаков паводка" if not flood else "переносный смысл"],
                          method=method)
    hits = {name: rx.search(text) for name, rx in (
        ("negation", NEGATION), ("historical", HISTORICAL), ("forecast", FORECAST),
        ("warning", WARNING), ("occurred", OCCURRED))}
    matched = [name for name, m in hits.items() if m]
    # Priority is conservative: a denial or an old story is never promoted to "it happened".
    report_type: ReportType = matched[0] if matched else "unclear"
    reasons = [f"правило: {name}" for name in matched] or ["нет однозначных признаков типа"]
    needs_review = report_type == "unclear" or len(matched) > 1
    quotes = [_quote(hits[name], text) for name in matched[:2]] or [_quote(flood, text)]
    return Extraction(relevant=True, report_type=report_type, place_mentions=places(text),
                      summary=title[:400], quotes=quotes, needs_review=needs_review,
                      reasons=reasons, method=method)
