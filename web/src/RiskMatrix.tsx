import { useState } from "react";
import type { Feature } from "./types";
import { finiteNumber, riskAsset } from "./riskAssets";
import { formatRub, formatProbability, formatNumber } from "./format";

export default function RiskMatrix({ assets, selected, onSelect, name, entropy }: {
  assets: Feature[]; selected: string | null; onSelect: (id: string) => void;
  name: (asset: Feature) => string; entropy: boolean;
}) {
  const [hovered, setHovered] = useState<string | null>(null);
  const points = assets.flatMap(feature => {
    const a = riskAsset(feature);
    return a ? [{ feature, ...a, loss: a.value * a.vulnerability }] : [];
  });
  const missing = assets.filter(a => !riskAsset(a));
  const max = Math.max(1, ...points.map(a => a.loss));
  const focus = points.find(p => p.id === (hovered ?? selected));
  return <section className="risk-matrix" aria-labelledby="risk-matrix-title">
    <div className="card-title"><h2 id="risk-matrix-title">Карта рисков объектов</h2></div>
    <p className="risk-intro">Вероятность воды и потери при её наличии. Выберите объект, чтобы связать оценку с таблицей и картой.</p>
    {points.length ? <>
      <svg viewBox="0 0 660 340" className="risk-plot" role="group" aria-label="Диаграмма: вероятность воды и условные потери">
        <text x="72" y="20" className="axis-title">Потери при наличии воды V × q, млн ₽</text>
        {[0, .25, .5, .75, 1].map(t => <g key={t}>
          <line x1={72 + t * 548} x2={72 + t * 548} y1="42" y2="274" />
          <line x1="72" x2="620" y1={274 - t * 232} y2={274 - t * 232} />
          <text x={72 + t * 548} y="298" textAnchor="middle">{t * 100} %</text>
          <text x="62" y={278 - t * 232} textAnchor="end">{new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 2 }).format(max * t / 1e6)}</text>
        </g>)}
        <text x="346" y="332" textAnchor="middle" className="axis-title">Вероятность воды p</text>
        {points.map(a => {
          const u = finiteNumber(a.feature.properties.uncertainty);
          const known = entropy && u != null && u >= 0 && u <= 1;
          const color = known ? `hsl(${195 - u * 155} 45% ${72 - u * 8}%)` : "#9ca9ae";
          const label = `${name(a.feature)} · ${a.id}: вероятность ${formatProbability(a.probability)}, условные потери ${formatRub(a.loss)}, ожидаемый ущерб ${formatRub(a.feature.properties.expected_loss_rub)}`;
          return <g key={a.id} role="button" tabIndex={0} aria-label={label} aria-pressed={selected === a.id}
            className={`risk-point ${selected === a.id ? "selected" : ""}`}
            transform={`translate(${72 + a.probability * 548}, ${274 - a.loss / max * 232})`}
            onClick={() => onSelect(a.id)} onMouseEnter={() => setHovered(a.id)} onMouseLeave={() => setHovered(null)}
            onFocus={() => setHovered(a.id)} onBlur={() => setHovered(null)}
            onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(a.id); } }}>
            <title>{label}</title><circle r="16" className="risk-point-hit" /><circle r="7" fill={color} />
            {(selected === a.id || hovered === a.id) && <text x={a.probability > .85 ? -12 : 12} y="-10" textAnchor={a.probability > .85 ? "end" : "start"}>{a.id}</text>}
          </g>;
        })}
      </svg>
      <div className="risk-legend">{entropy ? <><i />Неопределённость карты: от 0 до 1 бита · серый — нет оценки</> : "Цвет нейтральный: показатель не является энтропией в битах."}</div>
      <div className="risk-object-list" aria-label="Выбрать объект на диаграмме">{points.map(a => <button key={a.id} aria-pressed={selected === a.id} onClick={() => onSelect(a.id)}>{a.id} · {name(a.feature)}</button>)}</div>
      <p className="risk-readout" aria-live="polite">{focus ? <><strong>{name(focus.feature)} · {focus.id}</strong> · p {formatProbability(focus.probability)} · V × q {formatRub(focus.loss)} · ожидаемый ущерб {formatRub(focus.feature.properties.expected_loss_rub)} · неопределённость {finiteNumber(focus.feature.properties.uncertainty) == null ? "не оценена" : formatNumber(focus.feature.properties.uncertainty, 3)}</> : "Потери при наличии воды отличаются от ожидаемого ущерба: последний учитывает вероятность."}</p>
    </> : <p className="risk-intro">Нет оценённых объектов для диаграммы.</p>}
    {missing.length > 0 && <div className="risk-missing"><strong>Вне диаграммы: недостаточно данных</strong><div>{missing.map(a => <button key={String(a.properties.asset_id)} onClick={() => onSelect(String(a.properties.asset_id))}>{name(a)} · {String(a.properties.asset_id)}</button>)}</div></div>}
  </section>;
}
