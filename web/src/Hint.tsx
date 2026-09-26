import { useLayoutEffect, useRef, useState } from "react";

// A small "i" next to a label: the explanation appears on hover or keyboard focus. Clicks are
// swallowed so a hint inside a <label> does not toggle its checkbox. When shown, the bubble is
// shifted to stay inside its panel (panels scroll, so anything outside them is clipped).
export default function Hint({ text, align = "start" }: { text: string; align?: "start" | "center" | "end" }) {
  const [open, setOpen] = useState(false);
  const [shift, setShift] = useState(0);
  const bubble = useRef<HTMLSpanElement>(null);
  useLayoutEffect(() => {
    if (!open || !bubble.current) {
      setShift(0);
      return;
    }
    const box = bubble.current.getBoundingClientRect();
    const frame = bubble.current.closest(".context-panel, .analysis-content, .search-panel")
      ?.getBoundingClientRect() ?? { left: 0, right: window.innerWidth };
    const pad = 10;
    const scale = box.width / (bubble.current.offsetWidth || box.width || 1);
    let dx = 0;
    if (box.right > frame.right - pad) dx = frame.right - pad - box.right;
    if (box.left + dx < frame.left + pad) dx = frame.left + pad - box.left;
    setShift(dx / (scale || 1));
  }, [open]);
  return (
    <span
      className={`hint hint-${align}${open ? " open" : ""}`}
      tabIndex={0}
      role="note"
      aria-label={text}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
      onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
      }}
    >
      i
      <span
        ref={bubble}
        className="hint-bubble"
        aria-hidden="true"
        style={{ translate: `${shift}px 0` }}
      >
        {text}
      </span>
    </span>
  );
}

// Plain-language explanations of the operator's labels (layers, plan, analysis).
export const HINTS = {
  s1: "Исходный радарный снимок Sentinel-1 (канал VH). Тёмное — гладкая поверхность, чаще всего вода. Это единственный вход модели.",
  probability: "Вероятность воды p в каждом пикселе после калибровки: тёмно-синий — скорее сухо, яркий — почти наверняка вода. Отсюда берётся p объектов для ущерба.",
  mask: "Ответ «вода / не вода»: пиксель закрашен, если p ≥ порога 0,37 (подобран на validation). Ущерб считается по p, а не по маске.",
  uncertainty: "Насколько модель сомневается: 0 — уверена (p около 0 или 1), 1 — не знает (p = 50 %). Высокая неопределённость — повод проверить место снимком или осмотром.",
  assets: "10 объектов портфеля оператора, по одному каждого типа. Расставлены случайно с фиксированным seed, без опоры на прогноз и метки.",
  zones: "Варианты дополнительной съёмки: контур, площадь и цена по ПП № 840 со сценарной ставкой (не тариф). Синие — выбранные стратегией C.",
  coverage: "Доля ожидаемого ущерба объектов, попавших в купленные зоны. Это не предотвращённый ущерб.",
  decisionRisk: "Доля цены ошибки решения R = V·q·min(p, 1−p), которую закрывают снимки: там, где снимок может изменить решение «ехать / не ехать». Сценарная оценка.",
  zonesInPlan: "Сколько зон дополнительной съёмки выбрала стратегия C на текущий бюджет.",
  dataCost: "Цена снимков плана C по ПП № 840 со сценарной ставкой Б. Именно эту сумму ограничивает бюджет.",
  expectedLoss: "EL = p × V × q: вероятность воды в точке объекта × стоимость объекта × доля ущерба при затоплении. Ожидание по карте воды, не фактические потери.",
  inspection: "Порядок выездов: сначала объекты, где вероятна вода или модель не уверена, по убыванию ожидаемого ущерба; затем остальные.",
  strategyA: "Ничего не покупать и решать по текущей карте. Бесплатно, но сомнения остаются.",
  strategyB: "Новый радарный снимок всего чипа по заранее объявленному правилу. Если дороже бюджета — показана только для сравнения.",
  strategyC: "Наш выбор: зоны, где снимок может изменить решение «ехать / не ехать», в пределах бюджета.",
} as const;
