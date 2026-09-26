// Analyst workspace for media and eyewitness messages (standalone until the globe integration).
// All user and publication text goes through textContent — never innerHTML.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const T = {
    report_type: { occurred: "произошло", forecast: "прогноз", warning: "предупреждение",
      negation: "отрицание", historical: "исторический материал", unclear: "неясно" },
    review_state: { new: "новое", needs_geo: "нужно уточнить место", ready: "готово к проверке",
      under_review: "на проверке", reviewed: "проверено", merged: "объединено", dismissed: "отклонено" },
    verification: { not_checked: "не проверено", consistent: "наблюдения согласуются с сообщением",
      discrepancy: "есть расхождение", insufficient_data: "недостаточно подходящих данных" },
    precision: { point: "точка", street: "улица", locality: "населённый пункт", district: "район",
      region: "регион", country: "страна", unknown: "неизвестно" },
    time_precision: { hour: "до часа", day: "до дня", several_days: "несколько дней", unknown: "неизвестно" },
    origin: { news: "СМИ", bulletin: "сводка", citizen: "очевидец" },
    source_state: { ok: "работает", delayed: "задержка", error: "ошибка", disabled: "отключён", never: "ещё не запускался" },
    suitability: { suitable: "подходит", too_early: "слишком рано", too_late: "слишком поздно",
      partial_cover: "частичное покрытие", unusable: "не покрывает место", unknown_time: "время события неизвестно" },
    stage: { found: "найден в каталоге", acquired: "получен", prepared: "подготовлен", analyzed: "проанализирован" },
    follow_up: { none: "нет", needs_imagery: "нужна съёмка / проверка снимком", merge: "объединить",
      split: "разделить", dismiss: "отклонить как нерелевантное" },
  };
  let map, eventLayer, reportLayer, sceneLayer;
  let selected = null, features = [], lastCard = null;

  function h(tag, props, ...children) {
    const el = document.createElement(tag);
    Object.entries(props || {}).forEach(([k, v]) => {
      if (v === undefined || v === null || v === false) return;
      if (k === "text") el.textContent = v;
      else if (k === "class") el.className = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    });
    children.flat().forEach((c) => { if (c !== null && c !== undefined && c !== false) el.append(c.nodeType ? c : String(c)); });
    return el;
  }
  // Leaflet inserts string tooltips as HTML: pass a text-only node instead.
  const tipNode = (text) => h("div", { class: "tip", text });
  const tag = (text, cls) => h("span", { class: "tag " + (cls || ""), text });
  const fmt = (iso) => (iso ? new Date(iso).toLocaleString("ru-RU", { timeZone: "UTC" }) + " UTC" : "—");
  const toast = (text) => { const t = $("toast"); t.textContent = text; t.classList.remove("hidden"); setTimeout(() => t.classList.add("hidden"), 4000); };

  async function api(path, body) {
    const opts = { credentials: "same-origin", headers: {} };
    if (body !== undefined) {
      opts.method = "POST";
      opts.headers["Content-Type"] = "application/json";
      opts.headers["X-FloodValue-Media"] = "1";
      opts.body = JSON.stringify(body);
    }
    const resp = await fetch(path, opts);
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) { const err = new Error(data.detail || `HTTP ${resp.status}`); err.status = resp.status; throw err; }
    return data;
  }
  async function act(path, body, okText) {
    try { await api(path, body); toast(okText); await refresh(); if (selected) await openCard(selected, false); }
    catch (e) { toast("Не выполнено: " + e.message); }
  }

  // --- start ------------------------------------------------------------------------------
  async function start() {
    try { await api("/api/media/whoami"); }
    catch (e) {
      $("login").classList.remove("hidden");
      if (e.status === 403) { $("login").querySelector("form").classList.add("hidden"); $("login-note").textContent = e.message; }
      return;
    }
    $("app").classList.remove("hidden");
    initMap();
    ["f-origin", "f-type", "f-state", "f-since", "f-demo"].forEach((id) => $(id).addEventListener("change", loadEvents));
    $("layer-reports").addEventListener("change", loadReports);
    $("layer-scenes").addEventListener("change", () => { if (lastCard) drawScenes(lastCard); });
    $("ingest").addEventListener("click", async () => {
      $("ingest").disabled = true;
      try { const r = await api("/api/media/ingest", {}); toast(r.runs.map((x) => `${x.source_id}: ${x.status}, новых ${x.items_new}`).join("; ") || "нет включённых источников"); }
      catch (e) { toast("Сбор: " + e.message); }
      finally { $("ingest").disabled = false; refresh(); }
    });
    $("monitoring").addEventListener("change", (e) => act("/api/media/monitoring", { enabled: e.target.checked }, e.target.checked ? "Мониторинг включён" : "Мониторинг выключен"));
    await refresh();
  }
  async function refresh() { await Promise.all([loadSources(), loadReports(), loadEvents()]); }

  function initMap() {
    if (typeof L === "undefined") { $("map").textContent = "Карта не загрузилась (нет доступа к CDN) — списки и карточки работают."; return; }
    map = L.map("map", { zoomAnimation: !reduced, fadeAnimation: !reduced }).setView([20, 20], 2);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "© участники OpenStreetMap" }).addTo(map);
    eventLayer = L.layerGroup().addTo(map);
    reportLayer = L.layerGroup().addTo(map);
    sceneLayer = L.layerGroup().addTo(map);
  }

  // --- sources ----------------------------------------------------------------------------
  async function loadSources() {
    const data = await api("/api/media/sources");
    $("monitoring").checked = data.monitoring_enabled;
    const box = $("sources");
    box.replaceChildren(...data.sources.map((s) => {
      const cls = { ok: "ok", error: "bad", delayed: "warn" }[s.state] || "";
      const lines = [h("div", {}, h("strong", { text: s.name }), " ", tag(T.origin[s.kind] || s.kind), tag(T.source_state[s.state], cls))];
      lines.push(h("div", { class: "muted small", text: `последняя успешная проверка: ${fmt(s.last_success_at)}; новые данные: ${fmt(s.last_new_data_at)}` }));
      if (s.state === "error") lines.push(h("div", { class: "error", text: `ошибка ${fmt(s.last_error_at)}: ${s.last_error}. Пустой список из-за ошибки — не «событий нет».` }));
      if (s.terms_status !== "checked") lines.push(h("div", { class: "muted small", text: "условия использования не проверены — источник не включается" }));
      else lines.push(h("button", { class: "btn small", text: s.enabled ? "Отключить" : "Включить",
        onclick: () => act(`/api/media/sources/${encodeURIComponent(s.source_id)}/enabled`, { enabled: !s.enabled }, "Готово") }));
      return h("div", { class: "small", style: null }, ...lines, h("hr", { class: "m0" }));
    }));
  }

  // --- eyewitness queue -------------------------------------------------------------------
  async function loadReports() {
    const data = await api("/api/media/reports?status=pending");
    $("reports-count").textContent = String(data.reports.length);
    if (reportLayer) reportLayer.clearLayers();
    $("reports").replaceChildren(...data.reports.map((r) => {
      const photos = h("div", { class: "thumbs" }, ...r.photos.map((id) => h("img", { src: `/api/media/attachments/${id}`, alt: "фото из обращения", loading: "lazy" })));
      const actions = h("div", { class: "toolbar" },
        h("button", { class: "btn small", text: "Принять → событие", onclick: () => act(`/api/media/reports/${r.report_id}/moderate`, { decision: "accept" }, "Принято в работу") }),
        selected ? h("button", { class: "btn small", text: `Привязать к #${selected}`, onclick: () => act(`/api/media/reports/${r.report_id}/moderate`, { decision: "accept", event_id: selected }, "Привязано") }) : null,
        h("button", { class: "btn small", text: "Отклонить", onclick: () => act(`/api/media/reports/${r.report_id}/moderate`, { decision: "reject" }, "Отклонено") }),
        h("button", { class: "btn small", text: "Спам", onclick: () => act(`/api/media/reports/${r.report_id}/moderate`, { decision: "spam" }, "Помечено как спам") }));
      if (reportLayer && $("layer-reports").checked && r.lat !== null) {
        L.circleMarker([r.lat, r.lon], { radius: 6, color: "#d9b36c", weight: 1.5, dashArray: "3 3", fillOpacity: 0 })
          .bindTooltip(tipNode(`Непроверенное обращение ${r.report_id}`)).addTo(reportLayer);
      }
      return h("li", { tabindex: "0" },
        h("div", {}, r.is_demo ? tag("ДЕМО", "demo") : null, tag(r.category_label), h("span", { class: "muted small", text: r.report_id })),
        h("div", { text: r.description }),
        h("div", { class: "muted small", text: `место: ${r.place_text || "—"}${r.lat !== null ? ` (${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}, ${T.precision[r.location_precision]})` : " (без координат)"}` }),
        h("div", { class: "muted small", text: `наблюдение: ${r.observed_at ? fmt(r.observed_at) + ", " + T.time_precision[r.observed_precision] : "время не указано"}; отправлено: ${fmt(r.submitted_at)}` }),
        photos, actions);
    }));
  }

  // --- events -----------------------------------------------------------------------------
  async function loadEvents() {
    const q = new URLSearchParams();
    if ($("f-origin").value) q.set("origin", $("f-origin").value);
    if ($("f-type").value) q.set("report_type", $("f-type").value);
    if ($("f-state").value) q.set("review_state", $("f-state").value);
    if ($("f-since").value) q.set("since", new Date($("f-since").value).toISOString().slice(0, 19) + "Z");
    q.set("include_demo", $("f-demo").checked ? "true" : "false");
    const data = await api("/api/media/events?" + q);
    features = data.features;
    if (eventLayer) eventLayer.clearLayers();
    const geo = [], nogeo = [];
    features.forEach((f) => {
      const p = f.properties;
      const li = h("li", { tabindex: "0", "aria-current": String(p.event_id === selected), onclick: () => openCard(p.event_id, true),
        onkeydown: (e) => { if (e.key === "Enter") openCard(p.event_id, true); } },
        h("div", {}, p.is_demo ? tag(p.is_replay ? "ДЕМО · REPLAY" : "ДЕМО", "demo") : null, tag(T.report_type[p.report_type]),
          ...p.source_kinds.split(",").filter(Boolean).map((k) => tag(T.origin[k]))),
        h("div", { text: p.title }),
        h("div", { class: "muted small", text: `${p.place_label || "место неизвестно"} · ${T.review_state[p.review_state]} · ${T.verification[p.verification_result]}` }));
      (f.geometry ? geo : nogeo).push(li);
      if (f.geometry && eventLayer) drawEvent(f);
    });
    $("events").replaceChildren(...(geo.length ? geo : [h("li", { class: "muted", text: "нет сообщений по фильтру" })]));
    $("events-nogeo").replaceChildren(...(nogeo.length ? nogeo : [h("li", { class: "muted", text: "нет" })]));
  }

  function drawEvent(f) {
    const p = f.properties;
    const dashed = ["forecast", "warning", "unclear"].includes(p.report_type);
    const color = p.report_type === "negation" || p.report_type === "historical" ? "#929da2" : dashed ? "#d9b36c" : "#e9edef";
    const tip = `${p.is_demo ? "ДЕМО · " : ""}${p.title}\n${p.place_label || ""} (${T.precision[p.geo_precision]})\n${p.event_time_start ? fmt(p.event_time_start) : "время неизвестно"}\n${p.source_kinds.split(",").map((k) => T.origin[k]).join(", ")} · ${T.review_state[p.review_state]}`;
    let layer;
    const coarse = ["district", "region", "country"].includes(p.geo_precision);
    if (f.geometry.type === "Point" && coarse) {
      const [lon, lat] = f.geometry.coordinates;
      layer = L.circle([lat, lon], { radius: p.geo_precision === "district" ? 15000 : 60000, color, weight: 1.5, dashArray: "6 6", fill: false });
    } else if (f.geometry.type === "Point") {
      const [lon, lat] = f.geometry.coordinates;
      layer = L.circleMarker([lat, lon], { radius: 7, color, weight: 2, dashArray: dashed ? "3 3" : null, fillColor: color, fillOpacity: dashed ? 0 : 0.35 });
    } else {
      layer = L.geoJSON(f.geometry, { style: { color, weight: 1.5, dashArray: "6 6", fill: false } });
    }
    layer.bindTooltip(tipNode(tip)).on("click", () => openCard(p.event_id, true)).addTo(eventLayer);
  }

  function zoomTo(geometry, precision) {
    if (!map || !geometry) return;
    const g = L.geoJSON(geometry);
    const b = g.getBounds();
    const coarse = { district: 9, region: 6, country: 5 }[precision];
    if (geometry.type === "Point") map.setView(b.getCenter(), coarse || 11, { animate: !reduced });
    else map.fitBounds(b, { animate: !reduced, maxZoom: 11 });
  }

  function drawScenes(card) {
    if (!sceneLayer) return;
    sceneLayer.clearLayers();
    if (!$("layer-scenes").checked) return;
    card.observations.forEach((m) => {
      if (!m.footprint) return;
      L.geoJSON(m.footprint, { style: { color: "#6f8f9f", weight: 1, dashArray: "2 4", fill: false } })
        .bindTooltip(tipNode(`Спутниковый снимок (не сообщение): ${m.observation_ref}\n${m.observation_start ? fmt(m.observation_start) : ""} · ${T.suitability[m.suitability]}`))
        .addTo(sceneLayer);
    });
  }

  // --- card -------------------------------------------------------------------------------
  async function openCard(id, zoom) {
    selected = id;
    const card = await api(`/api/media/events/${id}`);
    lastCard = card;
    const e = card.event;
    if (zoom) zoomTo(e.geometry, e.geo_precision);
    drawScenes(card);
    document.querySelectorAll("#events li, #events-nogeo li").forEach((li) => li.setAttribute("aria-current", "false"));
    const pubTimes = card.evidence.map((x) => x.published_at).filter(Boolean).sort();
    const root = $("card");
    const evidenceRows = card.evidence.map((x) => {
      const check = h("input", { type: "checkbox", "data-evidence": x.evidence_id, "aria-label": "выделить в отдельное событие" });
      if (x.report) {
        return h("tr", {}, h("td", {}, check), h("td", {}, tag("очевидец")),
          h("td", {}, h("div", { text: x.report.description }), h("div", { class: "thumbs" }, ...(x.photos || []).map((pid) => h("img", { src: `/api/media/attachments/${pid}`, alt: "фото" })))),
          h("td", { class: "small", text: `наблюдение: ${x.report.observed_at ? fmt(x.report.observed_at) : "—"}; отправлено: ${fmt(x.report.submitted_at)}` }),
          h("td", {}, tag(T.report_type[x.report_type] || "—")));
      }
      const link = x.url && /^https?:\/\//.test(x.url) ? h("a", { href: x.url, target: "_blank", rel: "noopener noreferrer", text: x.title }) : h("span", { text: x.title });
      const basis = x.time_hint && x.time_hint.time_basis === "gdelt_seendate" ? " (время обнаружения GDELT)" : "";
      return h("tr", {}, h("td", {}, check), h("td", {}, tag(T.origin[x.origin]), x.is_original ? null : tag("перепечатка", "warn")),
        h("td", {}, link, h("div", { class: "muted small", text: `${x.source_name || ""}${x.source_domain ? " · " + x.source_domain : ""}${x.attribution ? " · " + x.attribution : ""}` }),
          x.text_excerpt ? h("div", { class: "small", text: x.text_excerpt }) : h("div", { class: "muted small", text: x.content_scope === "headline_only" ? "сохранён только заголовок" : "" }),
          x.facts && x.facts.quotes ? h("div", { class: "muted small", text: "фрагменты: " + x.facts.quotes.join(" … ") }) : null,
          h("div", { class: "muted small", text: `извлечение: ${x.extraction_method || "—"} · ${x.extraction_status || ""}` })),
        h("td", { class: "small", text: `публикация: ${fmt(x.published_at)}${basis}; получено системой: ${fmt(x.fetched_at)}` }),
        h("td", {}, tag(T.report_type[x.report_type] || "—")));
    });
    const obsRows = card.observations.map((m) => h("tr", {},
      h("td", { text: m.catalog === "cdse" ? "Copernicus Data Space" : "Sen1Floods11 (локально)" }),
      h("td", { class: "small", text: m.observation_ref }), h("td", { text: m.observation_start ? fmt(m.observation_start) : "—" }),
      h("td", {}, tag(T.suitability[m.suitability], m.suitability === "suitable" ? "ok" : "warn")),
      h("td", {}, tag(T.stage[m.stage]), m.analysis_run_id ? h("div", { class: "muted small", text: "комплект: " + m.analysis_run_id }) : null),
      h("td", { class: "small", text: m.reason })));
    const canConfirm = card.observations.some((m) => m.suitability === "suitable" && m.stage === "analyzed");

    root.replaceChildren(...[
      h("div", {}, e.is_demo ? tag(e.is_replay ? "ДЕМО · REPLAY (текст вымышлен)" : "ДЕМО", "demo") : null,
        tag(T.report_type[e.report_type]), tag(T.review_state[e.review_state]), tag(T.verification[e.verification_result])),
      h("h2", { class: "m0", text: `#${e.event_id} ${e.title}` }),
      ["forecast", "warning"].includes(e.report_type) ? h("p", { class: "tag warn", text: "Прогноз/предупреждение — не сообщение о произошедшем затоплении" }) : null,
      e.report_type === "negation" ? h("p", { class: "tag", text: "Сообщение-отрицание: само по себе событие не подтверждает и не опровергает" }) : null,
      h("div", { class: "toolbar" },
        h("button", { class: "btn small", text: "Источники", onclick: () => $("sec-sources").scrollIntoView({ behavior: reduced ? "auto" : "smooth" }) }),
        h("button", { class: "btn small", text: "Открыть территорию", disabled: !e.geometry, onclick: () => zoomTo(e.geometry, e.geo_precision) }),
        h("button", { class: "btn small", text: "Найти снимки", onclick: () => findScenes(e.event_id, false) }),
        h("button", { class: "btn small", text: "Найти снимки + Copernicus", title: "внешний каталог, только поиск записей", onclick: () => findScenes(e.event_id, true) }),
        h("button", { class: "btn small primary", text: "Проверить сообщение", onclick: () => $("sec-review").scrollIntoView({ behavior: reduced ? "auto" : "smooth" }) })),
      h("dl", { class: "kv" },
        h("dt", { text: "Где" }), h("dd", { text: `${e.place_label || "неизвестно"} · точность: ${T.precision[e.geo_precision]} · ${e.geo_status === "resolved" ? "определено" : e.geo_status === "ambiguous" ? "неоднозначно" : "не определено"}` }),
        h("dt", { text: "Когда произошло" }), h("dd", { text: e.event_time_start ? `${fmt(e.event_time_start)} — ${fmt(e.event_time_end)} (${T.time_precision[e.time_precision]})` : "неизвестно" }),
        h("dt", { text: "Первая публикация" }), h("dd", { text: fmt(pubTimes[0]) }),
        h("dt", { text: "Узнала система" }), h("dd", { text: fmt(e.first_detected_at) }),
        h("dt", { text: "Независимых первоисточников" }), h("dd", { text: `${e.independent_origin_count} (перепечатки не считаются; это не оценка достоверности)` }),
        h("dt", { text: "Что проверено" }), h("dd", { text: T.verification[e.verification_result] + (card.reviews.length ? ` — ${card.reviews[card.reviews.length - 1].reviewer}, ${fmt(card.reviews[card.reviews.length - 1].reviewed_at)}` : "") }),
        h("dt", { text: "Дальнейшее действие" }), h("dd", { text: T.follow_up[e.follow_up] })),
      h("p", { class: "section-title", text: "Каких данных не хватает" }),
      h("ul", { class: "small" }, ...card.missing.map((m) => h("li", { text: m }))),
      e.geo_candidates ? h("div", {}, h("p", { class: "section-title", text: "Варианты места (выберите, система не выбирает наугад)" }),
        ...e.geo_candidates.map((c) => h("button", { class: "btn small", text: `${c.label} (${T.precision[c.precision]})`,
          onclick: () => act(`/api/media/events/${e.event_id}/place`, { label: c.label, lon: c.lon, lat: c.lat, precision: c.precision }, "Место уточнено") }))) : null,
      h("p", { class: "section-title", id: "sec-sources", text: "Источники и свидетельства" }),
      h("table", {}, h("thead", {}, h("tr", {}, ...["", "тип", "сообщение", "время", "что сообщает"].map((t) => h("th", { text: t })))), h("tbody", {}, ...evidenceRows)),
      h("div", { class: "toolbar" }, h("button", { class: "btn small", text: "Выделить отмеченные в отдельное событие", onclick: () => {
        const ids = Array.from(root.querySelectorAll("input[data-evidence]:checked")).map((c) => Number(c.dataset.evidence));
        if (!ids.length) { toast("Отметьте свидетельства"); return; }
        act(`/api/media/events/${e.event_id}/split`, { evidence_ids: ids }, "Разделено");
      } })),
      card.possible_duplicates.length ? h("div", {}, h("p", { class: "section-title", text: "Возможные дубли (то же название места)" }),
        ...card.possible_duplicates.map((d) => h("div", { class: "small" }, `#${d.event_id} ${d.title} · ${T.report_type[d.report_type]} `,
          h("button", { class: "btn small", text: `Объединить #${d.event_id} → сюда`, onclick: () => act("/api/media/events/merge", { source_id: d.event_id, target_id: e.event_id }, "Объединено") })))) : null,
      h("p", { class: "section-title", text: "Спутниковые наблюдения" }),
      card.observations.length ? h("table", {}, h("thead", {}, h("tr", {}, ...["каталог", "снимок", "время съёмки", "пригодность", "этап", "почему"].map((t) => h("th", { text: t })))), h("tbody", {}, ...obsRows))
        : h("p", { class: "muted small", text: "Не искали или не найдено. Отсутствие снимка не опровергает сообщение." }),
      h("p", { class: "muted small", text: "Найденная запись каталога — не проверка. Вероятность воды появляется только после анализа подходящего снимка существующим конвейером FloodValue (вручную)." }),
      reviewForm(e, canConfirm), placeForm(e), timeForm(e), mergeForm(e),
      h("details", {}, h("summary", { class: "section-title", text: "История изменений" }),
        h("table", {}, h("tbody", {}, ...card.history.map((x) => h("tr", {}, h("td", { class: "small", text: fmt(x.at) }), h("td", { class: "small", text: x.actor }),
          h("td", { class: "small", text: x.action }), h("td", { class: "small muted", text: `${x.before_json || ""} → ${x.after_json || ""}` }))))))].filter(Boolean));
    loadReports();
  }

  function reviewForm(e, canConfirm) {
    const sel = h("select", { id: "rv-result" }, ...Object.entries(T.verification).map(([v, t]) =>
      h("option", { value: v, text: t, disabled: (v === "consistent" || v === "discrepancy") && !canConfirm })));
    sel.value = e.verification_result;
    const fu = h("select", { id: "rv-follow" }, ...Object.entries(T.follow_up).map(([v, t]) => h("option", { value: v, text: t })));
    fu.value = e.follow_up;
    const basis = h("input", { type: "text", id: "rv-basis", placeholder: "какие источники и снимки смотрели" });
    const comment = h("textarea", { id: "rv-comment" });
    return h("section", { class: "panel", id: "sec-review" }, h("h2", { text: "Проверить сообщение" }),
      canConfirm ? null : h("p", { class: "muted small", text: "Нет подходящего проанализированного снимка: доступны только «не проверено» и «недостаточно данных». Одобрение источника — не доказательство затопления." }),
      h("label", { for: "rv-result", text: "Результат сопоставления" }), sel,
      h("label", { for: "rv-follow", text: "Дальнейшее действие" }), fu,
      h("label", { for: "rv-basis", text: "Основание" }), basis,
      h("label", { for: "rv-comment", text: "Комментарий" }), comment,
      h("div", { class: "toolbar" }, h("button", { class: "btn primary small", text: "Сохранить проверку", onclick: () =>
        act(`/api/media/events/${e.event_id}/review`, { verification_result: sel.value, follow_up: fu.value, basis: basis.value, comment: comment.value }, "Проверка сохранена") })));
  }
  function placeForm(e) {
    const label = h("input", { type: "text", id: "pl-label", value: e.place_label || "" });
    const lat = h("input", { type: "number", id: "pl-lat", step: "any" }), lon = h("input", { type: "number", id: "pl-lon", step: "any" });
    const prec = h("select", { id: "pl-prec" }, ...Object.entries(T.precision).map(([v, t]) => h("option", { value: v, text: t })));
    prec.value = e.geo_precision === "unknown" ? "locality" : e.geo_precision;
    return h("details", { class: "panel" }, h("summary", { text: "Уточнить место" }),
      h("label", { for: "pl-label", text: "Название" }), label,
      h("div", { class: "row" }, h("div", {}, h("label", { for: "pl-lat", text: "Широта" }), lat), h("div", {}, h("label", { for: "pl-lon", text: "Долгота" }), lon)),
      h("label", { for: "pl-prec", text: "Точность" }), prec,
      h("div", { class: "toolbar" }, h("button", { class: "btn small", text: "Сохранить место", onclick: () =>
        act(`/api/media/events/${e.event_id}/place`, { label: label.value, lat: lat.value, lon: lon.value, precision: prec.value }, "Место сохранено") })));
  }
  function timeForm(e) {
    const s = h("input", { type: "datetime-local", id: "tm-start" }), en = h("input", { type: "datetime-local", id: "tm-end" });
    const prec = h("select", { id: "tm-prec" }, ...Object.entries(T.time_precision).map(([v, t]) => h("option", { value: v, text: t })));
    prec.value = e.time_precision;
    const iso = (v) => (v ? new Date(v).toISOString() : null);
    return h("details", { class: "panel" }, h("summary", { text: "Уточнить время события (время вашего устройства)" }),
      h("div", { class: "row" }, h("div", {}, h("label", { for: "tm-start", text: "Начало" }), s), h("div", {}, h("label", { for: "tm-end", text: "Конец" }), en)),
      h("label", { for: "tm-prec", text: "Точность" }), prec,
      h("div", { class: "toolbar" }, h("button", { class: "btn small", text: "Сохранить время", onclick: () =>
        act(`/api/media/events/${e.event_id}/time`, { start: iso(s.value), end: iso(en.value), precision: prec.value }, "Время сохранено") })));
  }
  function mergeForm(e) {
    const target = h("input", { type: "number", id: "mg-target", placeholder: "номер события" });
    return h("details", { class: "panel" }, h("summary", { text: "Объединить с другим событием" }),
      h("label", { for: "mg-target", text: "Перенести это событие в событие №" }), target,
      h("div", { class: "toolbar" }, h("button", { class: "btn small", text: "Объединить", onclick: () =>
        act("/api/media/events/merge", { source_id: e.event_id, target_id: Number(target.value) }, "Объединено") })));
  }
  async function findScenes(id, external) {
    try {
      const r = await api(`/api/media/events/${id}/observations`, { external });
      toast(`Найдено записей: ${r.found}` + (r.errors.length ? `; ${r.errors.join("; ")}` : ""));
      await openCard(id, false);
    } catch (e) { toast("Поиск: " + e.message); }
  }

  start();
})();
