// «Разбор СМИ»: one publication = one row through the six stages. Text goes in only as text.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const T = {
    report_type: { occurred: "произошло", forecast: "прогноз", warning: "предупреждение",
      negation: "опровержение", historical: "исторический материал", unclear: "неясно" },
    precision: { point: "точка", street: "улица", locality: "населённый пункт", district: "район",
      region: "регион", country: "страна", unknown: "неизвестно" },
    time_precision: { hour: "до часа", day: "до дня", several_days: "несколько дней", unknown: "неизвестно" },
    status: { ok: "о паводке", needs_review: "о паводке, нужна проверка", irrelevant: "не о паводке",
      invalid: "ответ модели отклонён", pending: "ещё не разобрана" },
    state: { ok: "работает", delayed: "задержка", error: "ошибка", never: "ещё не запускался", disabled: "выключен" },
    run: { ok: "успех", failed: "ошибка", skipped_rate_limit: "пропуск: лимит запросов",
      skipped_budget: "пропуск: бюджет", interrupted: "прерван", running: "идёт" },
    method: { rules: "правила", source_structured: "структура источника" },
  };
  // Same flood vocabulary as extract.FLOOD: shows the jury which words the filter caught.
  const FLOOD = /(?<!\p{L})(flood\p{L}*|inundat\p{L}*|submerg\p{L}*|deluge|наводнен\p{L}*|паводк\p{L}*|паводок|подтоп\p{L}*|затоп\p{L}*)/giu;

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "text") el.textContent = v; else if (k === "class") el.className = v; else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false)
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const fmt = (iso) => iso ? new Date(iso).toLocaleString("ru-RU", { dateStyle: "short", timeStyle: "short" }) : "—";
  const parse = (v) => { if (!v || typeof v === "object") return v || null; try { return JSON.parse(v); } catch { return null; } };
  function toast(text) {
    const t = $("toast"); t.textContent = text; t.classList.remove("hidden");
    clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.add("hidden"), 6000);
  }
  async function api(url, body) {
    const r = await fetch(url, body === undefined ? {} : { method: "POST",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(typeof data.detail === "string" ? data.detail : `Ошибка ${r.status}`), { status: r.status });
    return data;
  }

  // Marks: flood words (filter) and extracted places/quotes (extraction), found case-insensitively.
  function marked(text, places) {
    if (!text) return [];
    const spans = [];
    for (const m of text.matchAll(FLOOD)) spans.push([m.index, m.index + m[0].length, "hit-flood"]);
    const low = text.toLowerCase();
    for (const p of places) {
      const q = String(p || "").trim().toLowerCase();
      if (q.length < 3) continue;
      let i = low.indexOf(q);
      while (i >= 0) { spans.push([i, i + q.length, "hit-place"]); i = low.indexOf(q, i + q.length); }
    }
    spans.sort((a, b) => a[0] - b[0]);
    const out = []; let at = 0;
    for (const [s, e, cls] of spans) {
      if (s < at) continue;
      if (s > at) out.push(text.slice(at, s));
      out.push(h("mark", { class: cls, text: text.slice(s, e) }));
      at = e;
    }
    if (at < text.length) out.push(text.slice(at));
    return out;
  }

  function stage(n, title, ...body) {
    return h("div", { class: "st", "data-n": n }, h("div", { class: "st-t", text: `${n} · ${title}` }), ...body);
  }
  const line = (label, value, cls) => h("div", { class: "st-l" + (cls ? " " + cls : "") },
    h("span", { class: "muted", text: label + " " }), value ?? "—");

  function row(p, sourceNames) {
    const ex = parse(p.extraction ?? p.extraction_json) || {};
    const geo = parse(p.geocode) || parse(p.place_hint ?? p.place_hint_json) || {};
    const places = ex.place_mentions || [geo.label || geo.name].filter(Boolean);
    const status = p.extraction_status || "pending";
    const dropped = status === "irrelevant" || status === "invalid";
    const time = ex.event_time || {};
    const eventId = p.event_id ?? null;

    const src = stage(1, "Источник",
      h("div", { class: "st-strong", text: p.source_name || sourceNames[p.source_id] || p.source_id }),
      p.attribution ? h("div", { class: "muted small", text: p.attribution }) : null,
      line("получено", fmt(p.fetched_at)),
      p.is_demo ? h("span", { class: "tag demo", text: "ДЕМО" }) : null);

    const pub = stage(2, "Публикация",
      h("div", { class: "st-title" }, ...marked(p.title, places)),
      p.text_excerpt ? h("p", { class: "st-excerpt" }, ...marked(p.text_excerpt, places))
        : h("p", { class: "muted small m0", text: "Хранится только заголовок (условия источника)." }),
      h("div", { class: "st-l" },
        p.url && /^https?:/.test(p.url) ? h("a", { href: p.url, target: "_blank", rel: "noopener noreferrer", text: p.source_domain || "открыть источник" }) : null,
        p.published_at ? h("span", { class: "muted", text: " · опубликовано " + fmt(p.published_at) }) : null));

    const reasons = (ex.reasons || []).join("; ");
    const filt = stage(3, "Фильтр",
      h("span", { class: "tag " + (dropped ? "bad" : status === "needs_review" ? "warn" : "ok"), text: T.status[status] || status }),
      reasons ? line("правило", reasons) : null,
      p.processing_error ? line("ошибка", p.processing_error, "bad") : null);

    const extr = dropped ? stage(4, "Извлечение", h("p", { class: "muted small m0", text: "Не выполняется: публикация отброшена фильтром." })) :
      stage(4, "Извлечение",
        line("тип", T.report_type[p.report_type || ex.report_type] || "—"),
        line("место", places.length ? places.join(", ") : "не найдено"),
        line("время", time.start ? `${fmt(time.start)}${time.end && time.end !== time.start ? " – " + fmt(time.end) : ""} (${T.time_precision[time.precision] || "—"})` : "не указано"),
        line("метод", T.method[p.extraction_method] || p.extraction_method || "—"),
        (ex.quotes || []).length ? h("blockquote", { class: "st-quote", text: ex.quotes[0] }) : null);

    const gname = geo.label || geo.place_label || geo.name || geo.display_name;
    const gprec = geo.precision || geo.geo_precision || p.geo_precision;
    const [glon, glat] = geo.geometry?.coordinates || [geo.lon, geo.lat];
    const geoSt = dropped ? stage(5, "Геокодирование", h("p", { class: "muted small m0", text: "—" })) :
      stage(5, "Геокодирование",
        gname ? h("div", { class: "st-strong", text: gname }) : h("div", { class: "muted", text: "место не определено" }),
        line("точность", T.precision[gprec] || "—"),
        Number.isFinite(glat) && Number.isFinite(glon) ? line("координаты", `${glat.toFixed(3)}, ${glon.toFixed(3)}`) : null,
        geo.note ? line("примечание", geo.note) : null);

    const evSt = stage(6, "Событие",
      eventId !== null ? h("div", {},
        h("a", { class: "st-strong", href: `/media/analyst#event=${encodeURIComponent(eventId)}`, text: `Событие №${eventId}` }),
        p.event_sources ? line("независимых источников", p.event_sources) : null,
        h("div", { class: "muted small", text: "на карте оператора — янтарная точка" }))
        : h("p", { class: "muted small m0", text: dropped ? "Не создаётся." : "Не привязано: нет места или ждёт проверки." }));

    return h("li", { class: "pl-item" + (dropped ? " dropped" : "") }, src, pub, filt, extr, geoSt, evSt);
  }

  let sourceNames = {};
  async function loadSources() {
    try {
      const s = await api("/api/media/sources");
      const list = $("sources"); list.textContent = "";
      const sel = $("source-filter"); const keep = sel.value;
      sel.length = 1;
      for (const x of s.sources) {
        sourceNames[x.source_id] = x.name;
        sel.append(h("option", { value: x.source_id, text: x.name }));
        const r = x.last_run;
        const err = x.last_error || r?.error;
        list.append(h("li", { class: "src src-" + x.state },
          h("span", { class: "dot", "aria-hidden": "true" }),
          h("div", {},
            h("div", { class: "st-strong", text: x.name }),
            h("div", { class: "muted small", text:
              `${T.state[x.state] || x.state}` +
              (r ? ` · последний запуск ${fmt(r.started_at)}: ${T.run[r.status] || r.status}` +
                (r.http_status && r.http_status !== 200 ? ` (HTTP ${r.http_status})` : "") +
                `, просмотрено ${r.items_seen}, новых ${r.items_new}, о паводке ${r.items_relevant}` : "") }),
            err && x.state === "error" ? h("div", { class: "small bad", text: err }) : null)));
      }
      sel.value = keep;
      $("src-note").textContent = s.monitoring_enabled ? "Сборщик включён и работает по расписанию." : "Фоновый сбор выключен.";
    } catch (e) {
      $("sources").textContent = "";
      $("sources").append(h("li", { class: "muted", text: e.status === 401 || e.status === 403 ? "Доступно аналитику: войдите на /media/analyst." : e.message }));
    }
  }

  async function loadFeed() {
    const q = new URLSearchParams({ limit: "60" });
    if ($("source-filter").value) q.set("source", $("source-filter").value);
    if ($("rejected").checked) q.set("status", "irrelevant,invalid");
    const feed = $("feed");
    try {
      const data = await api("/api/media/publications?" + q);
      const pubs = data.publications || data.items || [];
      feed.textContent = "";
      if (!pubs.length) feed.append(h("li", { class: "muted", text: $("rejected").checked ? "Отброшенных публикаций нет." : "Публикаций пока нет — нажмите «Собрать сейчас»." }));
      for (const p of pubs) feed.append(row(p, sourceNames));
      $("count").textContent = pubs.length ? `показано ${pubs.length}` : "";
    } catch (e) {
      feed.textContent = "";
      feed.append(h("li", { class: "muted", text: e.status === 404 ? "Лента публикаций ещё не подключена на сервере." : e.message }));
    }
  }

  $("ingest").addEventListener("click", async () => {
    $("ingest").disabled = true; $("ingest").textContent = "Собираем…";
    try {
      const r = await api("/api/media/ingest", {});
      toast(r.runs.map((x) => `${sourceNames[x.source_id] || x.source_id}: ${T.run[x.status] || x.status}, новых ${x.items_new}`).join("; ") || "Нет включённых источников.");
    } catch (e) { toast(e.message); }
    finally { $("ingest").disabled = false; $("ingest").textContent = "Собрать сейчас"; await loadSources(); await loadFeed(); }
  });
  $("rejected").addEventListener("change", loadFeed);
  $("source-filter").addEventListener("change", loadFeed);

  loadSources().then(loadFeed);
  setInterval(() => { if (!document.hidden) loadSources(); }, 30000);
})();
