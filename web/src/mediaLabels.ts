// Russian labels for the media layer (/api/media/*): the operator never sees raw enum keys.
export const REPORT_TYPE: Record<string, string> = {
  occurred: "произошло", forecast: "прогноз", warning: "предупреждение",
  negation: "опровержение", historical: "исторический материал", unclear: "неясно",
};
export const REVIEW_STATE: Record<string, string> = {
  new: "новое", needs_geo: "нужно уточнить место", ready: "готово к проверке",
  under_review: "на проверке", reviewed: "проверено", merged: "объединено", dismissed: "отклонено",
};
export const VERIFICATION: Record<string, string> = {
  not_checked: "по снимкам не проверено", consistent: "снимки согласуются",
  discrepancy: "снимки расходятся", insufficient_data: "данных для проверки не хватает",
};
export const ORIGIN: Record<string, string> = { news: "СМИ", bulletin: "сводка", citizen: "очевидец" };
export const PRECISION: Record<string, string> = {
  point: "точка", street: "улица", locality: "населённый пункт", district: "район",
  region: "регион", country: "страна", unknown: "неизвестно", coarsened: "≈ 1 км",
};
export const MODERATION: Record<string, string> = {
  pending: "ждёт модерации", accepted: "принято в работу", rejected: "не принято", spam: "спам",
};
export const SUITABILITY: Record<string, string> = {
  suitable: "подходит", too_early: "снят до события", too_late: "снят слишком поздно",
  partial_cover: "покрывает частично", unusable: "непригоден", unknown_time: "время съёмки неизвестно",
};
export const EXTRACTION: Record<string, string> = {
  source_structured: "структурированные поля источника", rules_v1: "правила по тексту (v1)",
};

// Time as precise as the source says: a day-level or coarsened time never looks like "00:00".
export const when = (value: unknown, precision: unknown, coarse: boolean) => {
  if (typeof value !== "string" || !value) return "время неизвестно";
  const byDay = precision === "day" || precision === "several_days" || precision === "unknown";
  const text = new Intl.DateTimeFormat("ru-RU", byDay
    ? { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }
    : { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" },
  ).format(new Date(value));
  return byDay ? text : `${coarse ? "≈ " : ""}${text} UTC`;
};

export const label = (map: Record<string, string>, key: unknown, fallback = "—") =>
  typeof key === "string" && key ? map[key] ?? key : fallback;
