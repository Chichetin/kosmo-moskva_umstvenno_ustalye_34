const nf0 = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 0 });
const nf2 = new Intl.NumberFormat("ru-RU", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

export const rub = (v: number | null | undefined) => (v == null ? "—" : `${nf0.format(v)} ₽`);
export const rub2 = (v: number | null | undefined) => (v == null ? "—" : `${nf2.format(v)} ₽`);
export const num = (v: number | null | undefined, d = 2) =>
  v == null ? "—" : new Intl.NumberFormat("ru-RU", { maximumFractionDigits: d, minimumFractionDigits: d }).format(v);
export const pct = (v: number | null | undefined, d = 0) => (v == null ? "—" : `${num(v * 100, d)} %`);

export function mln(v: number | null | undefined) {
  if (v == null) return "—";
  if (Math.abs(v) >= 1e6) return `${num(v / 1e6, 2)} млн ₽`;
  if (Math.abs(v) >= 1e3) return `${num(v / 1e3, 1)} тыс. ₽`;
  return rub(v);
}

export function dt(iso: string | null | undefined) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString("ru-RU", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }) + " UTC";
}

export const ASSET_RU: Record<string, string> = {
  warehouse: "Склад",
  substation: "Электроподстанция",
  road_unit: "Участок дороги",
  facility: "Производственный объект",
  pumping_station: "Насосная станция",
  clinic: "Клиника",
  school: "Школа",
  workshop: "Мастерская",
  telecom_node: "Узел связи",
  water_intake: "Водозабор",
};

export const ACQ_RU: Record<string, string> = { new: "новая съёмка", operational: "оперативный архив", archive: "архив > 90 дн." };
export const STATUS_RU: Record<string, string> = {
  baseline: "исходная",
  scenario: "сценарий",
  measured: "измерена",
  not_estimated: "не оценена",
};
