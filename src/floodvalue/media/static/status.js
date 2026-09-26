// The token lives in the URL fragment: it is never sent in a request line or a Referer.
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  // Public status texts (media/reports.py PUBLIC_STATUS) → stage on the track; unknown → plain list.
  // Stage 2 is the analyst's decision either way: «Не принято в работу» is a final stage, not an error.
  const STAGE = { "На проверке у аналитика": 1, "Принято в работу": 2, "Не принято в работу": 2 };
  function track(status, submitted) {
    const stage = STAGE[status];
    if (stage === undefined) return false;
    const rows = [["Отправлено", submitted], ["На проверке у аналитика", ""],
      [stage === 2 ? status : "Решение аналитика", ""]];
    rows.forEach(([title, note], i) => {
      const li = document.createElement("li");
      const now = i === stage;  // stage 2 is final: its step is both reached and current
      li.className = (i < stage || (now && stage === 2) ? "done " : "") + (now ? "now" : "");
      if (now) li.setAttribute("aria-current", "step");
      const text = document.createElement("span");
      text.textContent = title;
      if (note) {
        const small = document.createElement("small");
        small.textContent = note;
        text.append(small);
      }
      li.append(text);
      $("track").append(li);
    });
    $("track").classList.remove("hidden");
    return true;
  }
  const token = location.hash.slice(1);  // secret-scan: allow — read from the URL, not a stored secret
  if (!/^[0-9a-f]{40}$/.test(token)) { $("state").textContent = "Ссылка неполная или неверная."; return; }
  try {
    const resp = await fetch("/api/media/reports/status", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
    if (!resp.ok) { $("state").textContent = resp.status === 404 ? "Обращение не найдено (могло быть удалено по сроку хранения)." : "Не удалось получить статус, попробуйте позже."; return; }
    const s = await resp.json();
    $("state").textContent = "Обращение";
    $("number").textContent = s.report_id;
    const submitted = new Date(s.submitted_at).toLocaleString("ru-RU");
    $("submitted").textContent = submitted;
    $("status").textContent = s.status;
    if (!track(s.status, submitted)) $("details").classList.remove("hidden");
  } catch (e) {
    $("state").textContent = "Нет связи — обновите страницу позже.";
  }
})();
