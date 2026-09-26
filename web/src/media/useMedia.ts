import { useCallback, useEffect, useState } from "react";
import { ApiError, opsGet, opsPost } from "./api";
import type { EventCard, MediaEvent, MessageFilters, Summary } from "./types";

const REFRESH_MS = 60_000;
export const defaultFilters: MessageFilters = {
  origins: { media: true, bulletin: true, citizen: true },
  days: 30,
  types: "situation",
  state: "open",
};

const query = (filters: MessageFilters) => {
  const params = new URLSearchParams({ days: String(filters.days) });
  for (const [origin, on] of Object.entries(filters.origins))
    if (on) params.append("origin", origin);
  if (filters.types === "situation")
    ["occurred", "unclear", "negation"].forEach((item) => params.append("report_type", item));
  if (filters.types === "alerts")
    ["forecast", "warning"].forEach((item) => params.append("report_type", item));
  if (filters.types === "all")
    ["occurred", "unclear", "negation", "forecast", "warning", "historical"].forEach((item) =>
      params.append("report_type", item),
    );
  if (filters.state === "open")
    ["new", "needs_geo", "ready", "under_review"].forEach((item) => params.append("review_state", item));
  if (filters.state === "needs_geo") params.append("review_state", "needs_geo");
  if (filters.state === "reviewed") params.append("review_state", "reviewed");
  return params.toString();
};

/** Operator session, message registry and the selected event card. */
export function useMedia(selectedEventId: number | null) {
  const [operator, setOperator] = useState<string | null | undefined>(undefined);
  const [filters, setFilters] = useState<MessageFilters>(defaultFilters);
  const [events, setEvents] = useState<MediaEvent[]>([]);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [card, setCard] = useState<EventCard | null>(null);
  const [cardError, setCardError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [showOnMap, setShowOnMap] = useState(true);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);

  useEffect(() => {
    opsGet<{ name: string }>("/api/ops/me")
      .then((me) => setOperator(me.name))
      .catch(() => setOperator(null));
  }, []);
  const onAuthError = (error: unknown) => {
    if (error instanceof ApiError && error.status === 401) setOperator(null);
  };

  useEffect(() => {
    if (!operator) return;
    const controller = new AbortController();
    setLoading(true);
    Promise.all([
      opsGet<MediaEvent[]>(`/api/ops/events?${query(filters)}`, controller.signal),
      opsGet<Summary>("/api/ops/summary", controller.signal),
    ])
      .then(([list, info]) => {
        setEvents(list);
        setSummary(info);
        setEventsError(null);
      })
      .catch((error) => {
        if (error.name === "AbortError") return;
        onAuthError(error);
        setEventsError(String(error.message));
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    const timer = window.setTimeout(refresh, REFRESH_MS);
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [operator, filters, revision, refresh]);

  useEffect(() => {
    if (!operator || selectedEventId == null) {
      setCard(null);
      setCardError(null);
      return;
    }
    const controller = new AbortController();
    opsGet<EventCard>(`/api/ops/events/${selectedEventId}`, controller.signal)
      .then((result) => {
        setCard(result);
        setCardError(null);
      })
      .catch((error) => {
        if (error.name === "AbortError") return;
        onAuthError(error);
        setCardError(String(error.message));
      });
    return () => controller.abort();
  }, [operator, selectedEventId, revision]);

  const login = async (token: string, name: string) => {
    const result = await opsPost<{ name: string }>("/api/ops/login", { token, name });
    setOperator(result.name);
  };
  const logout = async () => {
    await opsPost("/api/ops/logout").catch(() => undefined);
    setOperator(null);
    setEvents([]);
    setCard(null);
  };
  return {
    operator,
    login,
    logout,
    filters,
    setFilters,
    events,
    eventsError,
    loading,
    summary,
    card,
    setCard,
    cardError,
    refresh,
    showOnMap,
    setShowOnMap,
  };
}

export type Media = ReturnType<typeof useMedia>;
