// Public eyewitness form. Draft survives a network error (localStorage, best effort);
// one idempotency key per draft, so a retry never creates a second report.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const DRAFT = "fv-report-draft-v1";
  const fields = ["description", "place_text", "lat", "lon", "location_precision",
    "observed_local", "observed_precision"];
  let draft = load();
  let marker = null, map = null;
  let shotUrls = [];
  const NS = "http://www.w3.org/2000/svg";
  const wave = (y) => `M2.5 ${y}q1.6-1.4 3.2 0t3.2 0t3.2 0t3.2 0t3.2 0t3.2 0`;
  const ICONS = {
    flooding: ["M4 11 12 4.5l8 6.5", "M6.5 9.3V14", "M17.5 9.3V14", wave(17.5), wave(21)],
    water_rising: ["M12 13V3.5", "M8 7.5 12 3.5l4 4", wave(17.5), wave(21)],
    threat: ["M10.3 4.2 2.8 17.5a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z",
      "M12 9.5v4.2", "M12 17.2h.01"],
    other: ["M20 15a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2z",
      "M8.5 10.5h.01", "M12 10.5h.01", "M15.5 10.5h.01"],
  };

  function load() {
    try { return JSON.parse(localStorage.getItem(DRAFT)) || {}; } catch (e) { return {}; }
  }
  function save() {
    fields.forEach((f) => { draft[f] = $(f).value; });
    const cat = document.querySelector("input[name=category]:checked");
    draft.category = cat ? cat.value : "";
    try { localStorage.setItem(DRAFT, JSON.stringify(draft)); } catch (e) { /* private mode */ }
  }
  function key() {
    if (!draft.key) {
      const bytes = new Uint8Array(16);
      crypto.getRandomValues(bytes);
      draft.key = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
      save();
    }
    return draft.key;
  }
  function pad(n) { return String(n).padStart(2, "0"); }
  function localNow() {
    const d = new Date();
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  }
  function withOffset(local) {
    // datetime-local has no zone: attach the browser's offset explicitly.
    const d = new Date(local);
    if (isNaN(d)) return null;
    const off = -d.getTimezoneOffset();
    const sign = off >= 0 ? "+" : "-";
    return `${local}:00${sign}${pad(Math.floor(Math.abs(off) / 60))}:${pad(Math.abs(off) % 60)}`;
  }
  function setError(name, text, info) {
    const el = $("err-" + name);
    if (el) { el.textContent = text || ""; el.classList.toggle("info", !!info); }
  }
  function clearErrors() {
    ["category", "description", "place", "time", "photos", "form"].forEach((n) => setError(n, ""));
    document.querySelectorAll("[aria-invalid]").forEach((el) => el.removeAttribute("aria-invalid"));
  }
  function setPoint(lat, lon) {
    $("lat").value = lat.toFixed(6);
    $("lon").value = lon.toFixed(6);
    if (map) {
      if (marker) marker.setLatLng([lat, lon]);
      else {
        marker = L.marker([lat, lon], { icon: L.divIcon({ className: "pin", html: "<span></span>",
          iconSize: [30, 40], iconAnchor: [15, 38] }), draggable: true, alt: "Отмеченное место" }).addTo(map);
        marker.on("dragend", () => { const p = marker.getLatLng(); setPoint(p.lat, p.lng); });
      }
    }
    save();
    steps();
  }

  function icon(paths) {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.setAttribute("focusable", "false");
    paths.forEach((d) => {
      const path = document.createElementNS(NS, "path");
      path.setAttribute("d", d);
      svg.append(path);
    });
    return svg;
  }
  // Visual only: a step gets a check once it has what the submit check asks for.
  function steps() {
    const cat = document.querySelector("input[name=category]:checked");
    const done = [
      !!cat && $("description").value.trim().length >= 10,
      ($("lat").value.trim() !== "" && $("lon").value.trim() !== "") || $("place_text").value.trim() !== "",
      $("observed_precision").value === "unknown" || $("observed_local").value !== "",
      ($("photos").files || []).length > 0,
    ];
    done.forEach((ok, i) => $("step" + (i + 1)).classList.toggle("is-done", ok));
    $("progress").style.transform = `scaleX(${done.slice(0, 3).filter(Boolean).length / 3})`;
  }
  function showPhotos() {
    shotUrls.forEach((u) => URL.revokeObjectURL(u));
    shotUrls = [];
    const box = $("shots");
    box.replaceChildren();
    const files = Array.from($("photos").files || []);
    files.slice(0, 6).forEach((f) => {
      const img = document.createElement("img");
      const url = URL.createObjectURL(f);
      shotUrls.push(url);
      img.src = url;
      img.alt = f.name;
      box.append(img);
    });
    if (files.length) {
      const clear = document.createElement("button");
      clear.type = "button";
      clear.className = "btn ghost";
      clear.textContent = "Убрать фото";
      clear.addEventListener("click", () => { $("photos").value = ""; showPhotos(); steps(); });
      box.append(clear);
    }
  }

  async function init() {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "";
    const off = -new Date().getTimezoneOffset() / 60;
    $("tz").textContent = `(часовой пояс устройства: ${tz}, UTC${off >= 0 ? "+" : ""}${off})`;
    let config;
    try {
      config = await (await fetch("/api/media/report-config")).json();
    } catch (e) {
      config = { accepting: true, categories: {
        flooding: "Подтопление или затопление", water_rising: "Вода быстро поднимается",
        threat: "Угроза подтопления", other: "Другое" }, max_photos: 3, max_photo_mb: 8 };
    }
    if (!config.accepting) { $("closed").classList.remove("hidden"); $("form").classList.add("hidden"); return; }
    const box = $("categories");
    Object.entries(config.categories).forEach(([value, text]) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio"; input.name = "category"; input.value = value;
      input.checked = draft.category === value;
      input.addEventListener("change", save);
      const chip = document.createElement("span");
      chip.className = "chip";
      const span = document.createElement("span");
      span.textContent = text;
      chip.append(icon(ICONS[value] || ICONS.other), span);
      label.append(input, chip);
      box.append(label);
    });
    $("photo-hint").textContent = `До ${config.max_photos} фото, JPEG, PNG или WebP, до ${config.max_photo_mb} МБ каждое.`;
    fields.forEach((f) => { if (draft[f] !== undefined) $(f).value = draft[f]; });
    if (!$("observed_local").value) $("observed_local").value = localNow();
    fields.forEach((f) => $(f).addEventListener("input", save));
    ["lat", "lon"].forEach((f) => $(f).addEventListener("change", () => {
      const lat = parseFloat($("lat").value), lon = parseFloat($("lon").value);
      if (isFinite(lat) && isFinite(lon)) setPoint(lat, lon);
    }));
    initMap();
    $("geolocate").addEventListener("click", () => {
      if (!navigator.geolocation) { setError("place", "Геолокация недоступна в этом браузере."); return; }
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          setPoint(pos.coords.latitude, pos.coords.longitude);
          if (map) map.setView([pos.coords.latitude, pos.coords.longitude], 15);
          setError("place", "Отмечено ваше текущее положение. Если вода в другом месте — перенесите точку.", true);
        },
        () => setError("place", "Не удалось определить положение — отметьте место на карте."),
        { enableHighAccuracy: false, timeout: 10000 });
    });
    $("clear-point").addEventListener("click", () => {
      $("lat").value = ""; $("lon").value = "";
      if (marker && map) { map.removeLayer(marker); marker = null; }
      save();
      steps();
    });
    $("photos").addEventListener("change", showPhotos);
    $("form").addEventListener("input", steps);
    $("form").addEventListener("change", steps);
    steps();
    $("form").addEventListener("submit", submit);
    $("another").addEventListener("click", () => { location.reload(); });
    $("copy-link").addEventListener("click", async () => {
      const url = new URL($("done-link").getAttribute("href"), location.href).href;
      try {
        await navigator.clipboard.writeText(url);
        $("copied").textContent = "Ссылка скопирована.";
      } catch (e) {
        $("copied").textContent = "Не удалось скопировать — нажмите «Открыть статус обращения» и добавьте страницу в закладки.";
      }
    });
  }

  function initMap() {
    if (typeof L === "undefined") { $("map-fallback").classList.remove("hidden"); $("minimap").classList.add("hidden"); return; }
    map = L.map("minimap", { zoomControl: true, attributionControl: true }).setView([55.75, 37.62], 4);
    map.attributionControl.setPrefix('<a href="https://leafletjs.com" target="_blank" rel="noopener noreferrer">Leaflet</a>');  // plain library credit, no flag icon
    // OSM tile policy requires a Referer; the page itself stays no-referrer (status tokens).
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18, attribution: "© участники OpenStreetMap",
      referrerPolicy: "strict-origin-when-cross-origin" }).addTo(map);
    map.on("click", (e) => setPoint(e.latlng.lat, e.latlng.lng));
    const lat = parseFloat($("lat").value), lon = parseFloat($("lon").value);
    if (isFinite(lat) && isFinite(lon)) { setPoint(lat, lon); map.setView([lat, lon], 13); }
  }

  async function submit(ev) {
    ev.preventDefault();
    clearErrors();
    save();
    const cat = document.querySelector("input[name=category]:checked");
    const desc = $("description").value.trim();
    const lat = $("lat").value.trim(), lon = $("lon").value.trim();
    let bad = false;
    if (!cat) { setError("category", "Выберите, что произошло."); bad = true; }
    if (desc.length < 10) { setError("description", "Опишите подробнее — хотя бы 10 символов."); $("description").setAttribute("aria-invalid", "true"); bad = true; }
    if ((lat === "") !== (lon === "")) { setError("place", "Нужны и широта, и долгота."); bad = true; }
    if (lat === "" && !$("place_text").value.trim()) { setError("place", "Отметьте место на карте или напишите его название."); bad = true; }
    const files = Array.from($("photos").files || []);
    if (files.length > 3) { setError("photos", "Не больше 3 фото."); bad = true; }
    if (bad) { $("err-form").textContent = "Проверьте отмеченные поля."; return; }

    const data = new FormData();
    data.append("category", cat.value);
    data.append("description", desc);
    data.append("place_text", $("place_text").value.trim());
    if (lat !== "") { data.append("lat", lat); data.append("lon", lon); }
    data.append("location_precision", lat !== "" ? $("location_precision").value
      : ($("location_precision").value === "point" ? "locality" : $("location_precision").value));
    const precision = $("observed_precision").value;
    data.append("observed_precision", precision);
    if (precision !== "unknown" && $("observed_local").value) {
      const iso = withOffset($("observed_local").value);
      if (!iso) { setError("time", "Непонятная дата."); return; }
      data.append("observed_at", iso);
      data.append("observed_tz", Intl.DateTimeFormat().resolvedOptions().timeZone || "");
    }
    data.append("idempotency_key", key());
    files.forEach((f) => data.append("photos", f, f.name));

    const btn = $("submit");
    btn.disabled = true;
    btn.textContent = "Отправляем…";
    try {
      const resp = await fetch("/api/media/reports", { method: "POST", body: data });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        const field = (body.field || "").split(".").pop();
        const map2 = { lat: "place", lon: "place", place_text: "place", observed_at: "time",
          observed_precision: "time", photos: "photos", description: "description", category: "category" };
        setError(map2[field] || "form", body.detail || `Ошибка ${resp.status}`);
        if (!map2[field]) $("err-form").textContent = body.detail || `Ошибка ${resp.status}`;
        else $("err-form").textContent = "Проверьте отмеченные поля.";
        return;
      }
      try { localStorage.removeItem(DRAFT); } catch (e) { /* ignore */ }
      $("form").classList.add("hidden");
      $("done").classList.remove("hidden");
      $("done-number").textContent = body.report_id;
      $("done-link").href = "/report/status#" + body.status_token;
      $("done").focus();
    } catch (e) {
      $("err-form").textContent = "Не удалось отправить — нет связи. Введённый текст сохранён, попробуйте ещё раз: повторная отправка не создаст дубль.";
    } finally {
      btn.disabled = false;
      btn.textContent = "Отправить сообщение";
    }
  }

  init();
})();
