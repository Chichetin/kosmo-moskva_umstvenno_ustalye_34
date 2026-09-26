import { formatRub, formatScenarioLabel } from "./format";
import type { Session } from "./types";

type Row = Session["sensitivity"][number];

function scenarioInputs(row: Row): Record<string, unknown> {
  try {
    const value: unknown = JSON.parse(row.changed_inputs_json);
    return value && typeof value === "object" && !Array.isArray(value)
      ? value as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function numeric(value: unknown): number | null {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function rubExact(value: unknown): string {
  const number = numeric(value);
  return number == null ? "—" : `${new Intl.NumberFormat("ru-RU", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(number)} ₽`;
}

function statusLabel(value: string): string {
  switch (value) {
    case "within_budget": return "В бюджете данных";
    case "counterfactual_over_budget": return "Выше бюджета данных";
    case "empty_plan": return "Зон нет";
    case "ok": case "recomputed": return "Рассчитан";
    case "no_data": return "Нет данных";
    default: return "Статус не указан";
  }
}

const planScenario = (row: Row) => row.strategy === "C" && /^(budget|rate)_/.test(row.scenario_id);
const portfolioScenario = (row: Row) => row.strategy === "A" && /^[pVq]_portfolio_x[0-9.]+$/.test(row.scenario_id);
const scenarioResult = (row: Row) => row.strategy === "A"
  ? scenarioInputs(row).portfolio_expected_loss_rub : row.covered_expected_loss_rub;

export default function SensitivityPanel({ rows, baseBudget, baseLoss }: {
  rows: Session["sensitivity"];
  baseBudget: number;
  baseLoss: number | null;
}) {
  const planRows = rows.filter(planScenario);
  const portfolioRows = rows.filter(portfolioScenario);
  return (
    <section className="analysis-card sensitivity sensitivity-panel">
      <div className="card-title">
        <h2>Проверка устойчивости</h2>
        <span>Готовые сценарии комплекта</span>
      </div>
      <p className="sensitivity-intro">
        Отправная точка расчётов — исходный бюджет {formatRub(baseBudget)}. Строки бюджета показывают его отдельные варианты; ввод бюджета в панели плана не меняет сохранённые сценарии.
      </p>
      <div className="sensitivity-section">
        <h3>Бюджет и ставка съёмки → план C</h3>
        <p>Охваченный ущерб — сумма оценок объектов в выбранных зонах, не предотвращённый ущерб. Полная стоимость включает данные, расчёт и проверку зон.</p>
        {planRows.length ? <div className="table-scroll"><table>
          <thead><tr><th>Что изменено</th><th className="numeric">Охваченный ожидаемый ущерб</th><th className="numeric">Полная стоимость решения</th><th>Состояние</th></tr></thead>
          <tbody>{planRows.map((row) => <tr key={`${row.scenario_id}-${row.strategy}`}>
            <td>{formatScenarioLabel(row)}</td>
            <td className="numeric">{rubExact(row.covered_expected_loss_rub)}</td>
            <td className="numeric">{rubExact(row.decision_cost_rub)}</td>
            <td>{statusLabel(row.result_status)}</td>
          </tr>)}</tbody>
        </table></div> : <p className="sensitivity-empty">Сценариев бюджета и ставки в комплекте нет.</p>}
      </div>
      <div className="sensitivity-section">
        <h3>Вероятность, стоимость и уязвимость → ущерб</h3>
        <p>При изменении p, V или q пересчитан общий ожидаемый ущерб портфеля. Карта воды и план закупки в этих сценариях не пересчитывались.</p>
        {portfolioRows.length ? <div className="table-scroll"><table>
          <thead><tr><th>Что изменено</th><th className="numeric">Ожидаемый ущерб портфеля</th><th className="numeric">К исходной оценке</th></tr></thead>
          <tbody>
            <tr className="sensitivity-baseline"><td>Исходная оценка</td><td className="numeric">{rubExact(baseLoss)}</td><td className="numeric">—</td></tr>
            {portfolioRows.map((row) => {
              const value = numeric(scenarioInputs(row).portfolio_expected_loss_rub);
              const difference = value == null || baseLoss == null ? null : Math.round((value - baseLoss) * 100) / 100;
              return <tr key={`${row.scenario_id}-${row.strategy}`}>
                <td>{formatScenarioLabel(row)}</td>
                <td className="numeric">{rubExact(value)}</td>
                <td className="numeric">{difference == null ? "—" : `${difference > 0 ? "+" : difference < 0 ? "−" : ""}${rubExact(Math.abs(difference))}`}</td>
              </tr>;
            })}
          </tbody>
        </table></div> : <p className="sensitivity-empty">Сценариев изменения ущерба в комплекте нет.</p>}
      </div>
      <details className="sensitivity-all">
        <summary>Все сценарии комплекта · {rows.length}</summary>
        <p>Здесь сохранены также сравнения стратегии B, сроков, отдельных объектов и плана C с неизменным составом зон.</p>
        <div className="table-scroll"><table>
          <thead><tr><th>Сценарий</th><th>Стратегия</th><th className="numeric">Результат</th><th className="numeric">Полная стоимость</th><th>Состояние</th></tr></thead>
          <tbody>{rows.map((row) => <tr key={`${row.scenario_id}-${row.strategy}`}>
            <td>{formatScenarioLabel(row)}<small>{row.scenario_id}</small></td>
            <td>{row.strategy}</td>
            <td className="numeric">{rubExact(scenarioResult(row))}<small>{row.strategy === "A" ? "общий ущерб" : "охваченный ущерб"}</small></td>
            <td className="numeric">{rubExact(row.decision_cost_rub)}</td>
            <td>{statusLabel(row.result_status)}</td>
          </tr>)}</tbody>
        </table></div>
      </details>
    </section>
  );
}
