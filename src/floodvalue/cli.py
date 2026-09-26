"""Single entry point: one command per stage (`python -m floodvalue <stage>`), see README.

Each stage lives in its own module and talks to the others only through files described
in docs/contracts.md. A stage that is not implemented yet exits with code 2 and names
its issue area, so `make all` shows exactly where the chain is broken.
"""

import argparse
import importlib

# stage -> (module, help). Module must expose `run(args) -> int`.
STAGES = {
    "data": ("floodvalue.data.fetch", "скачать нужные чипы Sen1Floods11, QC, манифест"),
    "split": ("floodvalue.data.split", "построить/проверить замороженный split"),
    "baseline": ("floodvalue.baseline.run", "подобрать порог baseline на train/val, метрики val"),
    "train": ("floodvalue.model.train", "обучить основной метод (локально или Colab)"),
    "calibrate": ("floodvalue.calibration.run", "калибровка и порог на validation"),
    "predict": ("floodvalue.model.predict", "p-растр и маска: --chip <id> или --s1 <любой S1 GeoTIFF>"),
    "assets": ("floodvalue.assets.run", "10 объектов, p, EL, неопределённость, приоритеты"),
    "prewater": ("floodvalue.prewater.run", "вода до события по архиву S1 → слой «новая вода» (доп. файлы, #149)"),
    "procure": ("floodvalue.procurement.run", "зоны, цена ПП 840, A/B/C: --budget <руб>"),
    "export": ("floodvalue.export.run", "собрать 12 обязательных файлов"),
    "validate": ("floodvalue.export.validate", "проверить 12 файлов по контрактам"),
    "app": ("floodvalue.app.run", "интерфейс оператора"),
    "eval-test": ("floodvalue.model.eval_test", "финальная оценка на test (только замороженная конфигурация)"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="floodvalue")
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, (_, help_text) in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--chip", help="chip_id, например Ghana_103272")
        p.add_argument("--budget", type=float, help="бюджет платных данных, руб.")
        p.add_argument("--out", help="каталог результатов (по умолчанию FLOODVALUE_OUTPUT_DIR)")
        if name == "predict":
            p.add_argument("--s1", help="любой 2-канальный S1 GeoTIFF (VV, VH, dB), например "
                                        "S1Weak вне HandLabeled; метка не читается")
        if name == "eval-test":
            p.add_argument("--part", choices=["dev", "validation", "test"], default="test",
                           help="часть split; test — только с FLOODVALUE_EVAL_TEST_CONFIRM")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module_name, _ = STAGES[args.stage]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        area = module_name.split(".")[1]
        print(f"stage '{args.stage}' not implemented yet ({module_name}); issue area:{area}")
        return 2
    return int(module.run(args) or 0)
