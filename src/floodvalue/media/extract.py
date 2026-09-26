"""Conservative rule-based extraction (method «rules-v1»): no language model is configured.

The text of a publication is untrusted data: nothing in it is executed or obeyed, and it
never yields coordinates, casualty numbers or flood extents. Output is validated by a
strict schema; anything doubtful is marked for manual review instead of being guessed.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

METHOD = "rules-v1 (ключевые слова, без языковой модели)"

RELEVANT = re.compile(
    r"паводк|паводок|подтоп|затоп|наводнен|половод|разлив\w* рек|уровень вод|прорыв\w* дамб"
    r"|подъ[её]м\w* (?:воды|уровня)"
    r"|\bflood|flooding|inundat|submerged|burst (?:its |their )?banks|dam (?:burst|breach)"
    r"|waterlogg|deluge", re.IGNORECASE)
METAPHOR = re.compile(
    r"flood(?:ed|s|ing)? (?:of|with) (?:calls|requests|messages|complaints|emails|orders|"
    r"tourists|refugees|migrants|applications|comments|offers|tears|memories)|flood the zone|"
    r"floodgates|flood(?:ed)? the market|floodlight|наводнил\w* (?:рынок|сеть|интернет|соцсети)",
    re.IGNORECASE)
NEGATION = re.compile(
    r"не зафиксирован|не зарегистрирован|подтоплени\w* нет|затоплени\w* нет|угрозы "
    r"(?:подтоплен|затоплен|паводк)\w* нет|не подтвердил|опроверг|фейк|вода ушла|спад\w* вод"
    r"|no (?:reports of )?flooding|not flooded|no flood|waters? (?:recede|receded|receding)"
    r"|flood (?:fears|threat) (?:ease|eased|over)|all clear|debunk|fake", re.IGNORECASE)
FORECAST = re.compile(
    r"прогноз|ожида\w+|возможн\w+|может (?:быть|произойти|начаться)|вероятн\w+|в ближайш"
    r"|forecast|expected to|could (?:see|bring|cause|flood|hit)|may (?:flood|cause|bring|hit)"
    r"|likely to|predicted|projected|at risk of|risk of flood", re.IGNORECASE)
WARNING = re.compile(
    r"предупрежд|оповещен|угроз\w* (?:подтоплен|затоплен|паводк)"
    r"|warning|\balert\b|flood watch|evacuation order|advisory", re.IGNORECASE)
HISTORICAL = re.compile(
    r"лет назад|давности|годовщин|в память|вспомина|истори\w+ (?:паводк|наводнен)|years ago|anniversary"
    r"|remember|history of|looking back|archive", re.IGNORECASE)
OCCURRED = re.compile(
    r"затопил|подтопил|подтоплен|затоплен|ушл\w* под воду|эвакуир|вода (?:зашла|поднялась)"
    r"|смыл|размыл|погиб|пострадал|перекрыт|flooded|floods? (?:hit|hits|swamp|kill|leave|"
    r"displace|batter|ravage|inundate|submerge)|evacuat|swept away|submerged|inundated|stranded"
    r"|rescued|killed|death toll|underwater|washed away|burst (?:its |their )?banks"
    r"|режим\w* (?:чс|чрезвычайн)|declares? (?:a )?(?:state of )?(?:emergency|disaster)"
    r"|disaster (?:declared|zone)|emergency declared", re.IGNORECASE)
YEAR = re.compile(r"\b(19\d\d|20\d\d)\b")

STOP_WORDS = {
    "The", "A", "An", "In", "On", "At", "By", "BBC", "News", "Live", "Update", "Updates",
    "Watch", "Video", "Photos", "Flood", "Floods", "Flooding", "Heavy", "Rain", "Rains",
    "Storm", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December", "МЧС", "РФ", "Президент", "Правительство",
}
RU_PLACE = re.compile(
    r"(?:(?<=\s)|^)(?:в|во|на|под|у)\s+(?:(?:посёлке|поселке|селе|деревне|городе|станице|хуторе"
    r"|микрорайоне)\s+)?((?:Республике|Республика)\s+[А-ЯЁ][а-яё-]+"
    r"|[А-ЯЁ][а-яё-]+(?:\s+(?:област\w*|кра[еяй]\w*|район\w*|округ\w*|[А-ЯЁ][а-яё-]+))?)")
EN_PLACE = re.compile(
    r"\b(?:in|across|near|hit|hits|hitting|swamp|swamps|batter|batters|lash|lashes|of|to|over"
    r"|for|around|throughout)\s+"
    r"((?:[A-Z][\w'’-]+)(?:\s+(?:do|de|da|del|of|and)\s+[A-Z][\w'’-]+|\s+[A-Z][\w'’-]+){0,4})")
TITLE_LEAD = re.compile(r"^((?:[A-ZА-ЯЁ][\w'’-]+)(?:\s+[A-ZА-ЯЁ][\w'’-]+){0,2})\s+[a-zа-яё]")
EN_LEAD = re.compile(r"^((?:[A-Z][\w'’-]+\s){1,3})(?:floods?|flooding)\b")


class Extraction(BaseModel):
    """Strict schema of what extraction may claim. No coordinates, no numbers of victims."""

    model_config = ConfigDict(extra="forbid")
    relevance: Literal["relevant", "irrelevant", "unclear"]
    report_type: Literal["occurred", "forecast", "warning", "negation", "historical",
                         "unclear"] | None
    confidence: Literal["high", "medium", "low"]
    toponyms: list[str] = Field(default_factory=list, max_length=5)
    # Title-initial capitalised words: accepted only if the geocoder finds a prominent place.
    weak_toponyms: list[str] = Field(default_factory=list, max_length=1)
    signals: list[str] = Field(default_factory=list)
    status: Literal["ok", "invalid", "needs_review"]
    method: str = METHOD


def _years_only_past(text: str, now: datetime) -> bool:
    years = [int(value) for value in YEAR.findall(text)]
    return bool(years) and max(years) <= now.year - 2


def classify(title: str, excerpt: str | None = None, *, now: datetime | None = None) -> Extraction:
    now = now or datetime.now(UTC)
    # Figurative uses («flooded with calls») are cut out before looking for flood signals.
    text = METAPHOR.sub(" ", f"{title}. {excerpt or ''}")
    if not RELEVANT.search(text):
        return Extraction(relevance="irrelevant", report_type=None, confidence="high",
                          status="ok", signals=["нет признаков паводка"])
    signals = [name for name, pattern in (("отрицание", NEGATION), ("исторический", HISTORICAL),
                                          ("прогноз", FORECAST), ("предупреждение", WARNING),
                                          ("произошло", OCCURRED)) if pattern.search(text)]
    if _years_only_past(text, now):
        signals.append("упомянуты только прошлые годы")
    toponyms = extract_toponyms(text)
    lead = TITLE_LEAD.match(title.strip())
    weak = [name for name in [_clean(lead.group(1))] if len(name) >= 3 and name not in toponyms] \
        if lead else []
    if "отрицание" in signals:
        kind, confidence = "negation", "medium"
    elif "исторический" in signals or "упомянуты только прошлые годы" in signals:
        kind, confidence = "historical", "medium"
    elif "произошло" in signals and not {"прогноз", "предупреждение"} & set(signals):
        kind, confidence = "occurred", "high" if toponyms else "medium"
    elif "предупреждение" in signals and "произошло" not in signals:
        kind, confidence = "warning", "medium"
    elif "прогноз" in signals and "произошло" not in signals:
        kind, confidence = "forecast", "medium"
    else:
        kind, confidence = "unclear", "low"
    status = "ok" if confidence == "high" and toponyms else "needs_review"
    return Extraction(relevance="relevant", report_type=kind, confidence=confidence,
                      toponyms=toponyms, weak_toponyms=weak, signals=signals, status=status)


def _clean(name: str) -> str:
    words = [word for word in name.strip(" ,.:;«»\"'").split() if word not in STOP_WORDS]
    return " ".join(words)


def extract_toponyms(text: str) -> list[str]:
    """Place names as plain strings, in order of appearance; never coordinates."""
    found: list[str] = []
    candidates = [match.group(1) for match in RU_PLACE.finditer(text)]
    lead = EN_LEAD.match(text)
    if lead:
        candidates.append(lead.group(1))
    candidates += [match.group(1) for match in EN_PLACE.finditer(text)]
    for raw in candidates:
        for part in re.split(r"\s+(?:and|и)\s+|,\s*", raw):
            name = _clean(part)
            if len(name) >= 3 and name[0].isupper() and name not in found:
                found.append(name)
    return found[:5]


RU_ENDINGS = (
    (r"(\w+)ской област\w*", r"\1ская область"), (r"(\w+)цкой област\w*", r"\1цкая область"),
    (r"(\w+)ском кра\w*", r"\1ский край"), (r"(\w+)ском район\w*", r"\1ский район"),
    (r"(\w+)ском округ\w*", r"\1ский округ"), (r"Республике (\w+)", r"Республика \1"),
)


def toponym_variants(name: str) -> list[str]:
    """Nominative guesses for an inflected Russian toponym (Nominatim has no morphology)."""
    for pattern, replacement in RU_ENDINGS:
        if re.fullmatch(pattern, name):
            return [re.sub(pattern, replacement, name)]
    if re.fullmatch(r"[А-ЯЁ][а-яё-]+", name):
        variants = []
        if name.endswith("е"):
            variants += [name[:-1], name[:-1] + "а"]
        elif name.endswith("и"):
            variants += [name[:-1] + "ь", name[:-1] + "а"]
        return (variants or [name])[:2]
    return [name]
