/** Operator API: the session is an HttpOnly cookie; actions carry a header forms cannot send. */
export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(
  url: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(url, {
    ...init,
    signal,
    credentials: "same-origin",
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...(init.method && init.method !== "GET"
        ? { "X-FloodValue-Action": "1" }
        : {}),
      ...init.headers,
    },
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok)
    throw new ApiError(
      typeof data.detail === "string" ? data.detail : `Ошибка ${response.status}`,
      response.status,
    );
  return data as T;
}

export const opsGet = <T>(url: string, signal?: AbortSignal) =>
  request<T>(url, {}, signal);
export const opsPost = <T>(url: string, body: unknown = {}) =>
  request<T>(url, { method: "POST", body: JSON.stringify(body) });
export const opsPatch = <T>(url: string, body: unknown) =>
  request<T>(url, { method: "PATCH", body: JSON.stringify(body) });

export const labels = {
  origin: { media: "СМИ", bulletin: "Сводки", citizen: "Очевидцы" },
  reportType: {
    occurred: "Сообщение о подтоплении",
    forecast: "Прогноз",
    warning: "Предупреждение",
    negation: "Опровержение",
    historical: "Исторический материал",
    unclear: "Неясно",
  },
  reviewState: {
    new: "Новое",
    needs_geo: "Нужно уточнить место",
    ready: "Готово к проверке",
    under_review: "На проверке",
    reviewed: "Проверено",
    merged: "Объединено",
    split: "Разделено",
    dismissed: "Отклонено",
  },
  verification: {
    not_checked: "Не проверено",
    consistent: "Наблюдения согласуются",
    discrepancy: "Есть расхождение",
    insufficient_data: "Недостаточно данных",
  },
  precision: {
    point: "точка",
    street: "улица",
    locality: "населённый пункт",
    district: "район",
    region: "регион",
    country: "страна",
    unknown: "неизвестно",
  },
  timePrecision: {
    hour: "до часа",
    day: "до дня",
    several_days: "несколько дней",
    unknown: "неизвестно",
  },
  suitability: {
    suitable: "подходит",
    too_early: "раньше события",
    too_late: "слишком поздно",
    partial_cover: "частичное покрытие",
    unusable: "непригодно",
  },
  stage: {
    found: "найдено в каталоге",
    retrieved: "получено",
    prepared: "подготовлено",
    analyzed: "проанализировано",
  },
} as const;

export const label = (group: keyof typeof labels, key: string | null | undefined) =>
  key == null
    ? "—"
    : ((labels[group] as Record<string, string>)[key] ?? key.replaceAll("_", " "));

export const when = (value: string | null | undefined, withTime = true) =>
  value
    ? new Intl.DateTimeFormat("ru-RU", {
        day: "numeric",
        month: "short",
        year: "numeric",
        ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
      }).format(new Date(value))
    : "неизвестно";
