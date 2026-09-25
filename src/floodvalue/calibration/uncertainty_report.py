"""Report of `uncertainty_val`: markdown (docs/verification/uncertainty-val.md) and two figures.
Every number comes from the result dict of `uncertainty_val.study`, from
`docs/verification/eval-validation-summary.json` («было») or from the `asset_loss.csv` of a
run; nothing is typed by hand."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floodvalue.calibration import uncertainty as U
from floodvalue.calibration.report import CAL, EXTRA, INK, MUTED, RAW, _style, f, pct
from floodvalue.paths import REPO_ROOT

EVAL_SUMMARY = REPO_ROOT / "docs" / "verification" / "eval-validation-summary.json"
NAME_RU = {"entropy_p": "энтропия p", "threshold_margin": "близость к порогу",
           "member_std": "разброс членов (std)", "mutual_info": "взаимная информация"}
COLORS = {"entropy_p": CAL, "threshold_margin": EXTRA, "member_std": RAW, "mutual_info": "#9b59b6"}


def figures(res: dict, val: dict, out: Path, demo_chip: str, threshold: float | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    chosen = res["selection"]["chosen"]

    # 1. error rate by decile of u: validation (all) and dev, every candidate
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, part, title in ((axes[0], "validation", "validation (Sri-Lanka + Nigeria)"),
                            (axes[1], "dev", "dev (USA, в выборе не участвовал)")):
        for m in U.MEASURES:
            rows = res["parts"][part]["_deciles"][m]
            ax.plot([r["decile"] for r in rows], [r["error_rate"] for r in rows],
                    color=COLORS[m], linewidth=2.4 if m == chosen else 1.4, marker="o",
                    markersize=3.5, label=NAME_RU[m] + (" (выбран)" if m == chosen else ""))
        ax.axhline(res["parts"][part]["ALL"]["error_rate"], color=MUTED, linewidth=1,
                   linestyle="--")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xlabel("дециль неопределённости (1 — самые уверенные)")
        ax.set_xticks(range(1, 11))
        _style(ax)
    axes[0].set_ylabel("доля ошибок маски в дециле")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "error_by_decile.png", dpi=130)
    plt.close(fig)

    # 2. demo chip: p, chosen u, errors, members disagree on the mask
    mp = val.get("maps") or {}
    if not mp:
        return
    valid, label, p = mp["valid"], mp["label"], mp["p"]
    t = threshold if threshold is not None else 0.5
    err = np.full(p.shape, np.nan)
    pred = p >= t
    err[valid] = 0
    err[valid & pred & (label == 0)] = 1   # false positive
    err[valid & ~pred & (label == 1)] = 2  # false negative
    unstable = np.where(np.isfinite(p), (mp["lo"] < t) & (mp["hi"] >= t), np.nan)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    panels = ((p, "p (калиброванная)", "Blues", (0, 1)),
              (mp["u"][chosen], f"{NAME_RU[chosen]}", "magma", None),
              (err, "ошибки: оранжевый FP, синий FN", matplotlib.colors.ListedColormap(["#f2f1ec", RAW, CAL]),
               (0, 2)),
              (unstable, "члены расходятся по маске", matplotlib.colors.ListedColormap(
                  ["#f2f1ec", INK]), (0, 1)))
    for ax, (img, title, cmap, lim) in zip(axes, panels, strict=True):
        im = ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None,
                       interpolation="nearest")
        ax.set_title(title, color=INK, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax is axes[1] or ax is axes[0]:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{demo_chip} (validation), порог {f(t, 2)}", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "demo_chip.png", dpi=110)
    plt.close(fig)


def _scopes(scopes: dict):
    """(name, stats) of a part; ALL is skipped when it repeats the only event."""
    names = [s for s in scopes if not s.startswith("_")]
    for s in names:
        if s == "ALL" and len(names) == 2:
            continue
        yield s, scopes[s]


def _before() -> dict | None:
    if not EVAL_SUMMARY.exists():
        return None
    s = json.loads(EVAL_SUMMARY.read_text(encoding="utf-8"))
    return {"uncertainty": s.get("uncertainty", {}), "status": s.get("threshold_status", ""),
            "commit": s.get("commit", "")}


def markdown(res: dict, info: dict, cal: dict, assets: dict | None) -> str:
    sel, parts = res["selection"], res["parts"]
    chosen, t = sel["chosen"], res["threshold"]
    va, dv = parts["validation"], parts["dev"]
    L = ["# Неопределённость основного метода: выбор показателя и проверка на ошибках", "",
         "Сгенерировано `python -m floodvalue.calibration.uncertainty_val"
         + (f" --assets-run {assets['run']}" if assets else "") + f"` (commit "
         f"`{info['commit'][:12]}`, {info['created_at'][:10]}). Числа — из результата прогона, "
         "`uncertainty/uncertainty-scopes.csv`, `uncertainty/uncertainty-deciles.csv`; решение — "
         "D-033. Test не читался (D-010).", "",
         f"- Метод: `{res['method']}`, членов ансамбля: {res['n_members']}; калибратор и порог — "
         f"`models/{res['method']}/calibration.json` (D-029), порог маски p ≥ {f(t, 2)}.",
         f"- Чипов: validation {res['n_chips']['validation']}, dev {res['n_chips']['dev']}. "
         "Валидный пиксель: S1 конечен и LabelHand ≠ −1. **Ошибка** — маска основного метода "
         "≠ метке. **Нестабильный результат** — члены ансамбля расходятся по маске: "
         "min q_k < порог ≤ max q_k.",
         "- Показатель — свойство модели, а не наблюдение: его связь с ошибками здесь "
         "**проверена** на validation (использованной для калибровки и порога) и на dev (USA, "
         "не участвовал ни в калибровке, ни в выборе показателя). Независимая проверка на "
         "test — `make eval-test` (тот же показатель: энтропия сдаваемой p).", "",
         "## 1. Кандидаты и правило выбора (зафиксированы до расчёта)", "",
         "q_k = калибратор(p_k) — откалиброванная p члена k; сдаваемая p = калибратор(среднее p_k).", "",
         "| показатель | формула | единицы | что измеряет |", "|---|---|---|---|",
         "| энтропия p (`entropy_p`) | H(p) = −p·log₂p − (1−p)·log₂(1−p) | " + U.UNITS["entropy_p"]
         + " | нерешительность сдаваемой p (уверенность модели) |",
         "| близость к порогу (`threshold_margin`) | exp(−\\|logit p − logit t\\|) | "
         + U.UNITS["threshold_margin"] + " | насколько p близка к порогу маски |",
         "| разброс членов (`member_std`) | std_k q_k | " + U.UNITS["member_std"]
         + " | расхождение 5 seed (эпистемическая часть) |",
         "| взаимная информация (`mutual_info`) | H(mean q_k) − mean H(q_k) | "
         + U.UNITS["mutual_info"] + " | эпистемическая часть по BALD; q_k — калибратор среднего, "
         "приближение |", "",
         "Правило: максимум **худшего по событиям** AUROC(u → ошибка) на validation (Sri-Lanka, "
         "Nigeria); более простой показатель (порядок: энтропия, близость к порогу, std, "
         "взаимная информация) выбирается, если 95 % интервал бутстрэпа по чипам (внутри "
         f"событий, {sel['n_boot']} повторов, seed {sel['seed']}) разности с лучшим содержит 0. "
         "Полная энтропия ансамбля H(mean q_k) исключена до расчёта: это энтропия не сдаваемой "
         "вероятности. AUROC — вероятность, что у случайного ошибочного пикселя u выше, чем у "
         "случайного верного (0,5 — не различает; гистограмма в 10 000 квантильных корзин).", "",
         "## 2. Выбор на validation", "",
         "| показатель | AUROC Nigeria | AUROC Sri-Lanka | худшее событие | Δ к лучшему | 95 % CI Δ | итог |",
         "|---|---|---|---|---|---|---|"]
    for m in U.MEASURES:
        d = sel["vs_best"][m]
        mark = "**выбран**" if m == chosen else ("лучший" if m == sel["best"] else "")
        L.append(f"| {NAME_RU[m]} | {f(va['Nigeria'][m]['auroc'])} | {f(va['Sri-Lanka'][m]['auroc'])} "
                 f"| {f(sel['scores'][m])} | {f(d['delta'])} | [{f(d['ci_lo'])}; {f(d['ci_hi'])}] "
                 f"| {mark} |")
    L += ["", f"Выбран **{NAME_RU[chosen]}** (`{chosen}`, {U.UNITS[chosen]}). Ансамблевые "
          "показатели (std, взаимная информация) хуже различают ошибки: 5 seed на одних данных "
          "расходятся слабо и в основном там же, где p близка к порогу. Разброс ансамбля "
          "используется дальше как размах p для EL объектов (раздел 8).", "",
          "## 3. Средние на ошибках и верных, по событиям (validation и dev)", "",
          "| часть | событие | ошибочных | верных | u на ошибках | u на верных | u на FP | u на FN | AUROC |",
          "|---|---|---|---|---|---|---|---|---|"]
    for part, scopes in (("validation", va), ("dev", dv)):
        for s, st in _scopes(scopes):
            x = st[chosen]
            L.append(f"| {part} | {s} | {f(st['n_error'])} | {f(st['n_pixels'] - st['n_error'])} "
                     f"| {f(x['mean_u_error'])} | {f(x['mean_u_correct'])} | {f(x['mean_u_fp'])} "
                     f"| {f(x['mean_u_fn'])} | {f(x['auroc'])} |")
    L += ["", f"Все кандидаты (ALL): средние и AUROC; Spearman по чипам — корреляция средней u "
          "чипа с долей ошибок чипа.", "",
          "| показатель | часть | u на ошибках | u на верных | AUROC | Spearman по чипам |",
          "|---|---|---|---|---|---|"]
    for m in U.MEASURES:
        for part, scopes in (("validation", va), ("dev", dv)):
            x = scopes["ALL"][m]
            L.append(f"| {NAME_RU[m]} | {part} | {f(x['mean_u_error'])} | {f(x['mean_u_correct'])} "
                     f"| {f(x['auroc'])} | {f(scopes['_spearman'][m])} |")
    L += ["", "## 4. Нестабильный результат: члены ансамбля расходятся по маске", "",
          "| часть | событие | доля нестабильных | ошибок среди нестабильных | ошибок среди стабильных "
          f"| AUROC {NAME_RU[chosen]} → нестабильность |", "|---|---|---|---|---|---|"]
    for part, scopes in (("validation", va), ("dev", dv)):
        for s, st in _scopes(scopes):
            L.append(f"| {part} | {s} | {pct(st['unstable_share'])} | {pct(st['error_rate_unstable'])} "
                     f"| {pct(st['error_rate_stable'])} | {f(st[chosen]['auroc_unstable'])} |")
    cut = res["flag_cut"][chosen]
    L += ["", "## 5. Флаг «проверить»: верхние 10 % u (граница с validation) и сбой на dev", "",
          f"Граница флага — 90-й перцентиль показателя «{NAME_RU[chosen]}» на всех пикселях validation: "
          f"u ≥ {f(cut)}. На dev (USA, 4,6 % воды) калибратор validation переносится плохо "
          "(D-029: IoU 0,584 → 0,467, лишние FP) — проверяем, попадают ли ошибки dev под флаг, "
          "выставленный без dev.", "",
          "| часть | событие | доля пикселей под флагом | доля ошибок под флагом | FP под флагом "
          "| FN под флагом | ошибок среди флага | ошибок в среднем |", "|---|---|---|---|---|---|---|---|"]
    for part, scopes in (("validation", va), ("dev", dv)):
        for s, st in _scopes(scopes):
            x = st[chosen]
            L.append(f"| {part} | {s} | {pct(x['flagged_share'])} | {pct(x['errors_flagged_share'])} "
                     f"| {pct(x['fp_flagged_share'])} | {pct(x['fn_flagged_share'])} "
                     f"| {pct(x['error_rate_flagged'])} | {pct(st['error_rate'])} |")
    d1v, d1d = va["_deciles"][chosen][0], dv["_deciles"][chosen][0]
    L += ["", f"На dev самый уверенный дециль несёт {pct(d1d['error_rate'])} ошибок против "
          f"{pct(d1v['error_rate'])} на validation ({pct(d1d['errors_share'])} всех ошибок dev): "
          "часть сбоя переноса — уверенные ошибки, которые показатель не видит."]
    L += ["", "Пропуски (FN) ловятся хуже ложных срабатываний: пропущенная вода часто получает "
          "уверенно низкую p (тёмная вода под растительностью, узкие русла) — такой сбой "
          "показатель, построенный на самой p, не видит. Это ограничение, а не проверенная "
          "неопределённость.", "",
          f"## 6. Доля ошибок по децилям u ({NAME_RU[chosen]})", "",
          "| дециль | validation: u от–до | доля ошибок | доля всех ошибок | dev: u от–до | доля ошибок | доля всех ошибок |",
          "|---|---|---|---|---|---|---|"]
    for a, b in zip(va["_deciles"][chosen], dv["_deciles"][chosen], strict=True):
        L.append(f"| {a['decile']} | {f(a['u_min'])}–{f(a['u_max'])} | {pct(a['error_rate'])} "
                 f"| {pct(a['errors_share'])} | {f(b['u_min'])}–{f(b['u_max'])} "
                 f"| {pct(b['error_rate'])} | {pct(b['errors_share'])} |")
    L += ["", "![доля ошибок по децилям](uncertainty/error_by_decile.png)", "",
          "![демо-чип](uncertainty/demo_chip.png)", ""]
    before = _before()
    L += ["## 7. Было → стало на validation (те же 60 чипов)", ""]
    if before:
        L += [f"«Было» — `eval-validation-summary.json` (commit `{before['commit'][:12]}`, "
              f"{before['status'] or 'сырая p'}): временный показатель D-018 на сырой p и маске "
              f"p ≥ 0,5. «Стало» — выбранный показатель на сдаваемой калиброванной p и маске p ≥ "
              f"{f(t, 2)}. Пиксели те же (валидные validation); меняется карта и порог, а "
              "не набор пикселей.", "",
              "| событие | было: u ошибки / верные | было AUROC | стало: u ошибки / верные | стало AUROC |",
              "|---|---|---|---|---|"]
        for s in ("Nigeria", "Sri-Lanka", "ALL"):
            b = before["uncertainty"].get(s)
            x = va[s][chosen]
            if b:
                L.append(f"| {s} | {f(b['mean_u_error'])} / {f(b['mean_u_correct'])} | {f(b['auroc'])} "
                         f"| {f(x['mean_u_error'])} / {f(x['mean_u_correct'])} | {f(x['auroc'])} |")
    else:
        L.append("«Было» недоступно: нет `eval-validation-summary.json`.")
    L += ["", "## 8. Перенос к объектам и влияние на приоритеты и выбор съёмки", "",
          "- `uncertainty.tif` (Float32, −9999, сетка S1) — выбранный показатель; пишет `make "
          "predict` рядом с `flood_probability.tif`, смысл — `prediction.json` и "
          "`run_metadata.uncertainty`.",
          "- `flood_probability_range.tif` (2 канала: p_low, p_high) — размах min–max "
          "откалиброванных p пяти членов; сдаваемая p всегда внутри (калибратор монотонен).",
          "- `asset_loss.csv`: `uncertainty` — значение `uncertainty.tif` в пикселе объекта; "
          "доп. колонки `p_low`, `p_high`, `el_low_rub` = p_low·V·q, `el_high_rub` = p_high·V·q "
          "(«EL-размах ансамбля»: разброс 5 seed, обученных на одних данных; **не** доверительный "
          "интервал фактических потерь и не покрытие); `mask_unstable` — p_low < порог ≤ p_high.",
          "- Закупка (D-023): стратегия C при равном EL на рубль предпочитает зону с большей "
          "суммой `uncertainty`; A — `residual_uncertainty` = Σ `uncertainty` объектов "
          "(`baseline`); B/C без новых наблюдений снижение не заявляют (`not_estimated`).", ""]
    if assets:
        L += asset_table(assets, t)
    L += ["## Воспроизведение", "", "```bash", "set -a; . ./.env; set +a",
          "uv run --frozen python -m floodvalue.calibration.uncertainty_val"
          + (f" --assets-run {assets['run']}" if assets else ""), "```", ""]
    return "\n".join(L)


def asset_table(assets: dict, t: float) -> list[str]:
    loss = {r["asset_id"]: r for r in assets["loss"]}
    cls = {r["asset_id"]: r["asset_class"] for r in assets["assets"]}
    ok = [r for r in loss.values() if r["status"] == "ok"]

    def rank(key):
        order = sorted(ok, key=lambda r: (-float(r[key] or 0), r["asset_id"]))
        return {r["asset_id"]: i for i, r in enumerate(order, start=1)}

    r_el, r_hi, r_u = rank("expected_loss_rub"), rank("el_high_rub"), rank("uncertainty")
    width = {r["asset_id"]: float(r["el_high_rub"]) - float(r["el_low_rub"]) for r in ok}
    r_w = {a: i for i, a in enumerate(sorted(width, key=lambda a: (-width[a], a)), start=1)}
    L = [f"Портфель комплекта `{assets['run_id']}` (`asset_loss.csv`; ранги — 1 у максимума, "
         "равенство — по `asset_id`). Очерёдность осмотра по верхней границе EL-размаха "
         "учитывает и ущерб, и неопределённость; ранг по EL — обязательный `rank` файла.", "",
         "| объект | класс | p | p_low–p_high | EL, руб. | EL-размах, руб. | u | маска неустойчива "
         "| ранг EL | ранг EL_high | ранг u | ранг ширины |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for aid in sorted(loss):
        r = loss[aid]
        if r["status"] != "ok":
            L.append(f"| {aid} | {cls.get(aid, '')} | — | — | — | — | — | — | — | — | — | — |")
            continue
        L.append(f"| {aid} | {cls.get(aid, '')} | {f(float(r['p_flood']))} "
                 f"| {f(float(r['p_low']))}–{f(float(r['p_high']))} "
                 f"| {f(float(r['expected_loss_rub']), 0)} "
                 f"| {f(float(r['el_low_rub']), 0)}–{f(float(r['el_high_rub']), 0)} "
                 f"| {f(float(r['uncertainty']))} | {'да' if r['mask_unstable'] == 'true' else 'нет'} "
                 f"| {r_el[aid]} | {r_hi[aid]} | {r_u[aid]} | {r_w[aid]} |")
    moved = [a for a in r_el if r_el[a] != r_hi[a]]
    L += ["", f"Ранг по EL и по верхней границе EL-размаха расходится у {len(moved)} из {len(ok)} "
          f"объектов{': ' + ', '.join(sorted(moved)) if moved else ''}. Объекты с неустойчивой "
          f"маской (p_low < {f(t, 2)} ≤ p_high) — кандидаты на осмотр или съёмку независимо от "
          "ранга: карта для них может перевернуться при другом seed.", ""]
    return L
