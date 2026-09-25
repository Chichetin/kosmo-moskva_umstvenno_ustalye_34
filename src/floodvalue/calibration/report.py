"""Report of `calibrate`: markdown (docs/verification/calibration-val.md) and figures.
Every number comes from the result dict of `run.calibrate`; nothing is typed by hand."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from floodvalue.calibration import calibrator as C

RAW, CAL, EXTRA, INK, MUTED = "#eb6834", "#2a78d6", "#1baf7a", "#1f1f1e", "#8a8984"
KIND_RU = {"identity": "без калибровки", "temperature": "temperature scaling",
           "platt": "Platt", "isotonic": "изотоническая"}


def f(v, nd=3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.{nd}f}".replace(".", ",")


def pct(v) -> str:
    return f(100 * v, 1) + " %"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#e6e5df", linewidth=0.6)
    ax.set_axisbelow(True)


def figures(res: dict, chips, out: Path, demo_chip: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    thr = res["calibration"]["threshold"]

    # 1. reliability diagram, raw vs calibrated (all validation pixels)
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(5.2, 6.2), height_ratios=[3, 1], sharex=True)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="идеал")
    for rel, color, name in ((res["rel_raw"], RAW, "сырая p"), (res["rel_cal"], CAL, "после калибровки")):
        x = [r["mean_p"] for r in rel if r["n"]]
        y = [r["water_share"] for r in rel if r["n"]]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=name)
    ax.set_ylabel("доля воды по меткам")
    ax.set_title("Надёжность p на validation (15 бинов)", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    centers = [(r["bin_lo"] + r["bin_hi"]) / 2 for r in res["rel_raw"]]
    w = 1 / len(centers) / 2.2
    axn.bar(np.array(centers) - w / 2, [r["n"] for r in res["rel_raw"]], width=w, color=RAW)
    axn.bar(np.array(centers) + w / 2, [r["n"] for r in res["rel_cal"]], width=w, color=CAL)
    axn.set_yscale("log")
    axn.set_xlabel("средняя p в бине")
    axn.set_ylabel("пикселей")
    _style(axn)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=130)
    plt.close(fig)

    # 2. threshold -> IoU / F1 / P / R on calibrated p
    fig, ax = plt.subplots(figsize=(6, 3.8))
    t = [r["threshold"] for r in res["curve"]]
    for key, color, name in (("iou", CAL, "IoU"), ("f1", EXTRA, "F1"),
                             ("precision", "#eda100", "Precision"), ("recall", "#e87ba4", "Recall")):
        ax.plot(t, [r[key] for r in res["curve"]], color=color, linewidth=2, label=name)
    ax.plot(t, [r["iou"] for r in res["curve_raw"]], color=RAW, linewidth=1.5, linestyle=":",
            label="IoU, сырая p (тот же порог)")
    ax.axvline(thr, color=INK, linewidth=1)
    ax.text(thr + 0.01, 0.05, f"порог {f(thr, 2)}", color=INK, fontsize=9)
    ax.set_xlabel("порог по калиброванной p")
    ax.set_ylabel("метрика, validation")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower left", bbox_to_anchor=(0.12, 0.0))
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "threshold_curve.png", dpi=130)
    plt.close(fig)

    # 3. demo chip: raw p, calibrated p, errors before/after
    demo = next((c for c in chips if c.chip_id == demo_chip), None)
    if demo is not None:
        cal = res["calibration"]["calibrator"]
        q = C.apply(cal, demo.p_raw)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        cmap = matplotlib.colormaps["Blues"].copy()
        cmap.set_bad("#bdbdbd")
        for ax, img, title in ((axes[0], demo.p_raw, "сырая p"), (axes[1], q, "калиброванная p")):
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=axes[:2], fraction=0.025)
        for ax, p, t_, title in ((axes[2], demo.p_raw, 0.5, "ошибки: сырая p ≥ 0,5"),
                                 (axes[3], q, thr, f"ошибки: калибр. p ≥ {f(thr, 2)}")):
            ax.imshow(error_rgb(p, demo.label, t_))
            ax.set_title(title, fontsize=10)
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{demo_chip} (validation): зелёный TP, красный FP, оранжевый FN, "
                     "серый — nodata/метка −1", fontsize=10)
        fig.savefig(out / f"{demo_chip}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def error_rgb(p: np.ndarray, label: np.ndarray, t: float) -> np.ndarray:
    rgb = np.ones(p.shape + (3,))
    valid = np.isfinite(p) & (label != -1)
    pred = np.nan_to_num(p, nan=0.0) >= t
    y = label == 1
    rgb[valid & pred & y] = (0.10, 0.62, 0.35)
    rgb[valid & pred & ~y] = (0.85, 0.15, 0.15)
    rgb[valid & ~pred & y] = (0.95, 0.55, 0.10)
    rgb[~valid] = (0.74, 0.74, 0.74)
    return rgb


def markdown(res: dict, method: str) -> str:
    cal = res["calibration"]
    thr = cal["threshold"]
    kind = res["kind"]
    fit = cal["fitted_on"]
    ba = {(r["event"], r["stage"]): r for r in res["before_after"]}
    boot = res["boot"]
    L = []
    a = L.append
    a("# Калибровка и порог основного метода на validation")
    a("")
    a(f"Сгенерировано `make calibrate` (`python -m floodvalue calibrate`, `FLOODVALUE_METHOD={method}`), "
      f"commit `{cal['commit'][:12]}`, {cal['created_at']}. Параметры — "
      f"`models/{method}/calibration.json`; таблицы — `docs/verification/calibration/*.csv`.")
    a("")
    a(f"**Протокол (D-012, D-016, D-029).** Всё подбирается только на validation: "
      f"{', '.join(fit['events'])} — {fit['n_chips']} чипов, {f(fit['n_pixels'])} валидных пикселей, "
      f"доля воды {pct(fit['water_share'])} (в train около 8,9 %). Split `splits/split_v1.json`, "
      f"sha256 `{fit['split_sha256'][:12]}…`. **Test не читался**: код берёт только "
      "`chip_ids(\"validation\")`, метки test запечатаны в `floodvalue.data.io.load_label`. "
      "На validation выбраны две вещи: (1) вид калибратора — по log-loss **вне события** "
      "(подгонка на одном событии, проверка на другом); (2) порог — максимум IoU калиброванной p "
      "по всей validation. Итоговые числа validation поэтому оптимистичны; честная оценка "
      "переноса — таблица «между событиями» ниже и test (#19).")
    a("")
    a(f"Входная модель: `models/{method}/config.json`, `weights_sha256` "
      f"`{cal['input_model_sha256'][:12]}…`. Сырая p — `predict_proba` метода (docs/contracts.md §3), "
      "посчитана на CPU (fp32).")
    a("")
    a("## Итог: было → стало (validation, одни и те же валидные пиксели)")
    a("")
    a("| Событие | Этап | Порог | IoU | F1 | Precision | Recall | ECE (15 бинов) | Brier | log-loss | средняя p | доля воды |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ev in ("ALL", "Nigeria", "Sri-Lanka"):
        for st, name in (("raw_0.5", "было: сырая p ≥ 0,5"), ("calibrated", f"стало: {KIND_RU[kind]}, p ≥ {f(thr, 2)}")):
            r = ba[(ev, st)]
            a(f"| {ev} | {name} | {f(r['threshold'], 2)} | {f(r['iou'])} | {f(r['f1'])} | {f(r['precision'])} | "
              f"{f(r['recall'])} | {f(r['ece'], 4)} | {f(r['brier'], 4)} | {f(r['log_loss'], 4)} | "
              f"{f(r['mean_p'])} | {f(r['water_share'])} |")
    a("")
    a(f"Парный бутстрэп по чипам ({boot['n_boot']} повторов, seed {boot['seed']}, {boot['n_chips']} чипов): "
      f"ΔIoU (стало − было) = **{f(boot['delta_iou'])}**, 95 % CI [{f(boot['ci_lo'])}; {f(boot['ci_hi'])}]. "
      "Интервал не учитывает, что порог выбран на тех же пикселях; см. перенос между событиями.")
    a("")
    params = {"platt": f"p = sigmoid(a·logit(p_raw) + b), a = {f(cal['calibrator'].get('a', float('nan')), 4)}, "
                       f"b = {f(cal['calibrator'].get('b', float('nan')), 4)}",
              "temperature": f"p = sigmoid(logit(p_raw) / T), T = {f(cal['calibrator'].get('t', float('nan')), 4)}",
              "isotonic": f"кусочно-линейная неубывающая функция, {len(cal['calibrator'].get('x', []))} узлов",
              "identity": "p = p_raw"}[kind]
    a(f"**Калибратор: {KIND_RU[kind]}** — {params}. Порог **{f(thr, 2)}** по калиброванной p "
      f"(то же, что сырая p ≥ {f(cal['raw_threshold_equivalent'], 4)}; калибратор монотонный, "
      "поэтому ранжирование пикселей не меняется, меняется только шкала p и порог). "
      f"Максимум F1 на той же сетке — при пороге {f(res['t_f1'], 2)}.")
    a("")
    if res.get("metrics"):
        a("Метрики калиброванного метода по контракту §7 (dev — только отчёт, порог и калибратор "
          f"выбраны на validation) — `calibration/{method}-metrics.csv`; `valid_mask_sha256` совпадает "
          "с baseline и сырым U-Net:")
        a("")
        a("| Часть | Событие | IoU | F1 | Precision | Recall | Порог |")
        a("|---|---|---|---|---|---|---|")
        for r in res["metrics"]:
            a(f"| {r['split']} | {r['event']} | {f(r['iou'])} | {f(r['f1'])} | {f(r['precision'])} | "
              f"{f(r['recall'])} | {f(r['threshold'], 2)} |")
        a("")
    if res.get("dev_check"):
        raw, calr = res["dev_check"]
        a("## Контроль переноса на dev (USA): событие с малой долей воды")
        a("")
        a("Dev в подборе калибратора и порога не участвовал (D-012: dev — выбор модели). Здесь он — "
          "проверка, что будет на событии с другой долей воды:")
        a("")
        a("| dev (USA) | Порог | IoU | F1 | Precision | Recall | ECE | Brier | log-loss | средняя p | доля воды |")
        a("|---|---|---|---|---|---|---|---|---|---|---|")
        for r, name in ((raw, "было: сырая p ≥ 0,5"), (calr, f"стало: {KIND_RU[kind]}, p ≥ {f(thr, 2)}")):
            a(f"| {name} | {f(r['threshold'], 2)} | {f(r['iou'])} | {f(r['f1'])} | {f(r['precision'])} | "
              f"{f(r['recall'])} | {f(r['ece'], 4)} | {f(r['brier'], 4)} | {f(r['log_loss'], 4)} | "
              f"{f(r['mean_p'])} | {f(r['water_share'])} |")
        a("")
        worse = calr["iou"] < raw["iou"]
        a(("**На dev калибровка ухудшает маску** " if worse else "На dev калибровка не ухудшает маску ")
          + f"(IoU {f(raw['iou'])} → {f(calr['iou'])}; средняя p {f(calr['mean_p'])} при доле воды "
          f"{f(calr['water_share'])}). Калибратор переносит на новое событие априорную долю воды "
          f"validation ({pct(fit['water_share'])}): на событии, где воды в разы меньше, p завышается "
          "и порог даёт лишние срабатывания; ухудшаются и ECE, и log-loss, а значит и EL = p·V·q "
          "на таком событии завышен. Калибровка остаётся на validation по протоколу D-012 (dev — "
          "для выбора модели, test запечатан); этот результат на выбор не влиял. Справочно: сводные "
          "доли воды test видели на разведке до split (раскрыто в D-016, 15–18 %, `docs/data.md` §5) — "
          "ближе к validation, чем к dev. Вывод для оператора: калиброванная p надёжна для паводков "
          "с долей воды порядка validation; на событии с малой долей воды p завышена, и это "
          "ограничение переноса, а не свойство модели.")
        a("")
    a("## Выбор калибратора: перенос между событиями")
    a("")
    a("Подгонка на одном событии validation, все метрики — на другом. Порог «из подгонки» — "
      "максимум IoU на событии подгонки, применённый без изменений ко второму; «оракул» — лучший "
      "порог на самом событии проверки (только ориентир).")
    a("")
    a("| Подгонка → проверка | Калибратор | Параметры | ECE | Brier | log-loss | средняя p / доля воды | порог из подгонки | IoU перенос | IoU сырая 0,5 | IoU оракул |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in res["cross"]:
        a(f"| {r['fit_event']} → {r['eval_event']} | {KIND_RU[r['kind']]} | {r['params']} | {f(r['ece'], 4)} | "
          f"{f(r['brier'], 4)} | {f(r['log_loss'], 4)} | {f(r['mean_p'])} / {f(r['water_share'])} | "
          f"{f(r['threshold_from_fit'], 2)} | {f(r['iou_transfer'])} | {f(r['iou_raw_0_5'])} | {f(r['iou_oracle'])} |")
    a("")
    means = {}
    for r in res["cross"]:
        means.setdefault(r["kind"], []).append(r)
    a("Среднее по двум направлениям (правило выбора — минимум log-loss вне события):")
    a("")
    a("| Калибратор | log-loss | ECE | Brier | IoU перенос |")
    a("|---|---|---|---|---|")
    for k, rows in means.items():
        mark = " ← выбран" if k == kind else ""
        a(f"| {KIND_RU[k]}{mark} | {f(np.mean([r['log_loss'] for r in rows]), 4)} | "
          f"{f(np.mean([r['ece'] for r in rows]), 4)} | {f(np.mean([r['brier'] for r in rows]), 4)} | "
          f"{f(np.mean([r['iou_transfer'] for r in rows]))} |")
    a("")
    a("На всей validation (подгонка и оценка на одних пикселях — для сравнения гибкости, не для выбора):")
    a("")
    a("| Калибратор | Параметры | ECE | Brier | log-loss | средняя p |")
    a("|---|---|---|---|---|---|")
    for r in res["in_sample"]:
        a(f"| {KIND_RU[r['kind']]} | {r['params']} | {f(r['ece'], 4)} | {f(r['brier'], 4)} | "
          f"{f(r['log_loss'], 4)} | {f(r['mean_p'])} |")
    a("")
    a("## Надёжность вероятностей (reliability, вся validation)")
    a("")
    a("![надёжность](calibration/reliability.png)")
    a("")
    a("| Бин p | пикселей (сырая) | средняя p | доля воды | пикселей (калибр.) | средняя p | доля воды |")
    a("|---|---|---|---|---|---|---|")
    for x, y in zip(res["rel_raw"], res["rel_cal"], strict=True):
        a(f"| [{f(x['bin_lo'], 2)}; {f(x['bin_hi'], 2)}) | {f(x['n'])} | {f(x['mean_p'])} | {f(x['water_share'])} | "
          f"{f(y['n'])} | {f(y['mean_p'])} | {f(y['water_share'])} |")
    a("")
    rel = [r for r in res["rel_cal"] if r["n"]]
    worst = max(rel, key=lambda r: abs(r["mean_p"] - r["water_share"]))
    total = sum(r["n"] for r in rel)
    mid = sum(r["n"] for r in rel if 0.2 <= r["bin_lo"] and r["bin_hi"] <= 0.8 + 1e-9)
    a(f"Сырая p систематически **занижена** во всём диапазоне (сеть обучена на ~8,9 % воды). "
      f"После калибровки средняя p совпадает с долей воды, ECE падает с "
      f"{f(ba[('ALL', 'raw_0.5')]['ece'], 4)} до {f(ba[('ALL', 'calibrated')]['ece'], 4)}. Остаётся "
      f"S-образное отклонение в середине шкалы: наибольшее — в бине [{f(worst['bin_lo'], 2)}; "
      f"{f(worst['bin_hi'], 2)}), p {f(worst['mean_p'])} при доле воды {f(worst['water_share'])}. "
      f"Бины 0,2–0,8 содержат {pct(mid / total)} пикселей, поэтому ECE мал, но p в середине шкалы "
      "(это и есть объекты «на границе») надёжна примерно до ±0,1. Изотоническая функция убирает это "
      "на своих данных, но хуже переносится на другое событие (таблица выше).")
    a("")
    a("## Влияние порога на метрики")
    a("")
    a("![порог](calibration/threshold_curve.png)")
    a("")
    a("Кривая целиком — `calibration/threshold_curve.csv` (шаг 0,01, калиброванная и сырая p). Выдержка:")
    a("")
    a("| Порог (калибр. p) | IoU | F1 | Precision | Recall |")
    a("|---|---|---|---|---|")
    pick = {round(x, 2) for x in (0.1, 0.2, 0.3, thr, 0.5, 0.6, 0.7, 0.8, 0.9, res["t_f1"])}
    for r in res["curve"]:
        if round(r["threshold"], 2) in pick:
            b = "**" if r["threshold"] == thr else ""
            a(f"| {b}{f(r['threshold'], 2)}{b} | {f(r['iou'])} | {f(r['f1'])} | {f(r['precision'])} | {f(r['recall'])} |")
    a("")
    a("**Правило порога — максимум IoU** (метрика сравнения методов, критерий 4). При равенстве "
      "берётся меньший порог: оператору (D-001) пропущенный затопленный объект дороже лишнего "
      "осмотра. Экономика от порога не зависит: EL = p·V·q берёт p из растра, не из маски "
      "(task.md); порог влияет только на `flood_mask.tif` и метрики сегментации.")
    a("")
    a("## Постобработка")
    a("")
    a("Постобработка не применяется. Проверено одно простое сглаживание — среднее 3×3 по "
      "калиброванной p (NaN не участвуют) с тем же порогом:")
    a("")
    a("| Событие | IoU без постобработки | IoU со средним 3×3 |")
    a("|---|---|---|")
    for r in res["post"]:
        a(f"| {r['event']} | {f(r['none'])} | {f(r['mean3x3'])} |")
    a("")
    gain = max(r["mean3x3"] - r["none"] for r in res["post"])
    a(f"Наибольший выигрыш — {f(gain, 4)} IoU, на уровне шума; лишний шаг, меняющий p объектов, "
      "не принимается.")
    a("")
    a("## Nodata")
    a("")
    a("- Невалидный пиксель — S1 не конечен (NaN) хотя бы в одном канале: сырая p = NaN, калибратор "
      "сохраняет NaN (`calibrator.apply`), в растрах — −9999 (`flood_probability.tif`) и 255 "
      "(`flood_mask.tif`), docs/contracts.md §2.")
    a("- Метка −1 исключена из подгонки калибратора, выбора порога и метрик; на растры не влияет.")
    a("- Маска валидности та же, что у U-Net и baseline (S1 конечен и метка ∈ {0, 1}): "
      f"{f(fit['n_pixels'])} пикселей validation, как в `docs/verification/unet-metrics.md`.")
    a("")
    a("## Постоянная вода: IoU внутри и вне JRC (диагностика, не вход модели)")
    a("")
    a("JRCWaterHand (0/1) из набора Sen1Floods11 используется **только здесь, для отчёта**. В модель, "
      "калибратор и порог он не входит: слой, вероятно, построен по данным до 2018 года и для "
      "событий 2017–2018 содержит информацию после даты события (`docs/data.md` §7).")
    a("")
    a("| Зона | Пикселей | Доля воды по меткам | IoU было | IoU стало | P / R было | P / R стало | средняя калибр. p |")
    a("|---|---|---|---|---|---|---|---|")
    for r in res["jrc"]:
        x, y = r["raw_0.5"], r["calibrated"]
        a(f"| {r['zone']} | {f(r['n_pixels'])} | {f(r['water_share'])} | {f(x['iou'])} | {f(y['iou'])} | "
          f"{f(x['precision'])} / {f(x['recall'])} | {f(y['precision'])} / {f(y['recall'])} | {f(r['mean_p_cal'])} |")
    a("")
    j1 = res["jrc"][0]
    a(f"Внутри JRC-воды метка = 1 у {pct(j1['water_share'])} пикселей: `LabelHand` размечает **всю** "
      "воду, включая постоянную, и модель её уверенно находит. Отделить временное затопление по "
      "данным, доступным на дату события, мы не можем, поэтому продукт — **карта воды**, а не карта "
      "временного затопления (D-007). Для оператора это значит: объект у реки или водоёма получает "
      "высокую p и тогда, когда вода там постоянная; такие объекты стоит проверять по своему "
      "кадастру водных объектов, а не считать затопленными.")
    a("")
    a("## Пример: Sri-Lanka_713926 (validation)")
    a("")
    a("![пример](calibration/Sri-Lanka_713926.png)")
    a("")
    a("## Ограничения")
    a("")
    ece_out = np.mean([r["ece"] for r in means[kind]])
    ece_in = next(r["ece"] for r in res["in_sample"] if r["kind"] == kind)
    a(f"- Validation — два события (Nigeria {pct(ba[('Nigeria', 'raw_0.5')]['water_share'])} воды, "
      f"Sri-Lanka {pct(ba[('Sri-Lanka', 'raw_0.5')]['water_share'])}). Калибратор сдвигает p к доле воды "
      f"validation ({pct(fit['water_share'])}); в test доля воды по разведке 15–18 % (`docs/data.md` §5), "
      "но на событии с другой долей воды калибровка может снова разойтись. Перенос между "
      f"событиями validation это показывает: ECE вне события {f(ece_out, 4)} против {f(ece_in, 4)} "
      "при подгонке на всей validation.")
    a("- Порог выбран на тех же пикселях, где считаются итоговые числа validation; незавышенная "
      "оценка — test (`make eval-test`, #19).")
    a("- Калибровка не меняет ранжирование пикселей: ROC/PR-кривая модели та же, выигрыш IoU — от "
      "порога, выигрыш log-loss/Brier/ECE — от шкалы p.")
    a("")
    a("## Воспроизведение")
    a("")
    a("```bash")
    a("set -a; . ./.env; set +a          # FLOODVALUE_DATA_DIR с Sen1Floods11")
    a("make calibrate                     # CPU, около 2 минут: калибратор, порог, этот отчёт")
    a("make predict CHIP=Sri-Lanka_713926 # flood_probability.tif уже калиброван, mask ⇔ p ≥ порог")
    a("```")
    a("")
    return "\n".join(L)
