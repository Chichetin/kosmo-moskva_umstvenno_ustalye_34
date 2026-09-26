// The token lives in the URL fragment: it is never sent in a request line or a Referer.
(async function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const token = location.hash.slice(1);
  if (!/^[0-9a-f]{40}$/.test(token)) { $("state").textContent = "Ссылка неполная или неверная."; return; }
  try {
    const resp = await fetch("/api/media/reports/status", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token }) });
    if (!resp.ok) { $("state").textContent = resp.status === 404 ? "Обращение не найдено (могло быть удалено по сроку хранения)." : "Не удалось получить статус, попробуйте позже."; return; }
    const s = await resp.json();
    $("state").textContent = "Обращение";
    $("number").textContent = s.report_id;
    $("submitted").textContent = new Date(s.submitted_at).toLocaleString("ru-RU");
    $("status").textContent = s.status;
    $("details").classList.remove("hidden");
  } catch (e) {
    $("state").textContent = "Нет связи — обновите страницу позже.";
  }
})();
