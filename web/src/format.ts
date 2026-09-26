function finite(value: unknown): number | null {
  if (typeof value !== "number" && typeof value !== "string") return null;
  if (typeof value === "string" && !value.trim()) return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatNumber(value: unknown, maximumFractionDigits = 0): string {
  const number = finite(value);
  return number == null
    ? "—"
    : new Intl.NumberFormat("ru-RU", { maximumFractionDigits }).format(number);
}

export function formatRub(value: unknown): string {
  const number = finite(value);
  return number == null ? "—" : `${formatNumber(number)} ₽`;
}

export function formatShare(value: unknown): string {
  const number = finite(value);
  return number == null ? "—" : `${formatNumber(number * 100, 1)} %`;
}

export function formatProbability(value: unknown): string {
  const number = finite(value);
  if (number == null) return "—";
  const percent = number * 100;
  if (percent > 0 && percent < 0.01) return "<0,01 %";
  return `${formatNumber(percent, 2)} %`;
}

export function formatExactProbability(value: unknown): string {
  const number = finite(value);
  return number == null ? "—" : String(number);
}

type SensitivityRow = Record<string, string>;

function changedInputs(row: SensitivityRow): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(row.changed_inputs_json);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed as Record<string, unknown>
      : {};
  } catch {
    return {};
  }
}

function factor(value: unknown): number | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  return finite((value as Record<string, unknown>).factor);
}

export function formatScenarioLabel(row: SensitivityRow): string {
  const changed = changedInputs(row);
  const budget = finite(changed.budget_rub);
  if (budget != null) return `Бюджет ${formatRub(budget)}`;

  const rate = finite(changed.base_rate_rub_km2);
  if (rate != null) return `Сценарная ставка съёмки ${formatNumber(rate)} ₽/км²`;

  const names: Array<[string, string, boolean]> = [
    ["p_flood", "Вероятности воды", true],
    ["asset_value_rub", "Стоимость объектов", false],
    ["vulnerability_coef", "Уязвимость объектов", true],
  ];
  for (const [key, name, bounded] of names) {
    const multiplier = factor(changed[key]);
    if (multiplier == null) continue;
    const scope = changed.scope === "single_asset" && typeof changed.asset_id === "string"
      ? ` · ${changed.asset_id}` : "";
    const change = multiplier === 0.5 ? "уменьшены вдвое"
      : `× ${formatNumber(multiplier, 2)}`;
    return `${name} ${change}${bounded && multiplier > 1 ? " (не выше 1)" : ""}${scope}`;
  }

  if (changed.rule_C === "by_EL") return "План C: ущерб на рубль стоимости данных";
  if (typeof changed.deadline_hours_after_decision === "number") {
    return `Срок решения через ${formatNumber(changed.deadline_hours_after_decision)} ч`;
  }
  return row.scenario_id.replaceAll("_", " ");
}

export function formatScenarioAmount(row: SensitivityRow): string {
  const changed = changedInputs(row);
  const value = row.strategy === "A"
    ? changed.portfolio_expected_loss_rub
    : row.covered_expected_loss_rub;
  return formatRub(value);
}
