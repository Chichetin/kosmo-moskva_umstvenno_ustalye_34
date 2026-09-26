"""The eyewitness pages may change their look freely, but not what the CSP allows, the safety texts,
or the element ids their scripts read."""

import re
from pathlib import Path

import pytest

from floodvalue.media.web import create_app

from .conftest import call

STATIC = Path(__file__).resolve().parents[2] / "src/floodvalue/media/static"
PAGES = {"/report": ("report.html", "report.js"), "/report/status": ("status.html", "status.js")}


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.mark.parametrize("path", PAGES)
def test_page_fits_its_csp(app, path):
    """style-src and script-src are 'self' + jsDelivr, without 'unsafe-inline'."""
    html = call(app, "GET", path).body.decode()
    assert " style=" not in html and "<style" not in html
    assert all("src=" in tag for tag in re.findall(r"<script[^>]*>", html))
    assert "/media/static/report.css" in html and "media.css" not in html  # media.css is the analyst's
    css = call(app, "GET", "/media/static/report.css")
    assert css.status == 200 and css.headers["content-type"].startswith("text/css")


def test_safety_texts_stay(app):
    report = call(app, "GET", "/report").body.decode()
    for text in ("Это не вызов экстренных служб.", "Если есть угроза жизни — звоните",
                 "Отправка не подтверждает событие и не заказывает спутниковую съёмку автоматически.",
                 "Не подходите к воде ради фотографии.", "Регистрация, имя и телефон не нужны.",
                 "Сообщение хранится ограниченный срок и видно только аналитикам.",
                 "Отправка не означает подтверждение события и не заказывает спутниковую съёмку."):
        assert text in report
    status = call(app, "GET", "/report/status").body.decode()
    assert "Это не подтверждение затопления спутником и не заказ съёмки." in status


@pytest.mark.parametrize("path", PAGES)
def test_script_ids_exist_in_page(path):
    html_name, js_name = PAGES[path]
    html = (STATIC / html_name).read_text(encoding="utf-8")
    js = (STATIC / js_name).read_text(encoding="utf-8")
    ids = set(re.findall(r'\$\("([\w-]+)"\)', js))
    fields = re.search(r"const fields = \[([^\]]*)\]", js)
    if fields:
        ids |= set(re.findall(r'"([\w-]+)"', fields.group(1)))
    errors = re.search(r"\[([^\]]*)\]\.forEach\(\(n\) => setError", js)
    if errors:
        ids |= {"err-" + n for n in re.findall(r'"([\w-]+)"', errors.group(1))}
    if "\"step\" + (i + 1)" in js:
        ids |= {f"step{i}" for i in range(1, 5)}
    missing = sorted(i for i in ids if f'id="{i}"' not in html)
    assert ids and not missing, missing


def test_page_fonts_are_served_with_licence(app):
    css = (STATIC / "report.css").read_text(encoding="utf-8")
    fonts = re.findall(r"/media/static/fonts/([\w-]+\.woff2)", css)
    assert fonts
    for name in fonts:
        r = call(app, "GET", f"/media/static/fonts/{name}")
        assert r.status == 200 and r.headers["content-type"] == "font/woff2", name
        assert (STATIC / "fonts" / (name.split("-")[0] + "-OFL.txt")).is_file(), name


def test_pages_follow_operator_visual_system():
    """#247: the public pages share the operator console's look — one typeface, dark only,
    corners only through the 2 px / 0 tokens (dots may be round), no decorative scene."""
    css = (STATIC / "report.css").read_text(encoding="utf-8")
    assert "--radius: 2px" in css and "--radius-panel: 0" in css
    assert "Science Gothic" in css and "prefers-color-scheme" not in css
    assert "gradient(" not in css
    shadows = re.findall(r"box-shadow:\s*([^;}]+)", css)
    assert all(s.strip().startswith(("none", "0 0 0")) for s in shadows), shadows  # rings only
    radii = re.findall(r"border-radius:\s*([^;!}]+)", css)
    assert radii and all(r.strip().startswith("var(--radius") or r.strip() == "50%" for r in radii), radii
    for name in ("report.html", "status.html"):
        html = (STATIC / name).read_text(encoding="utf-8")
        assert 'class="scene"' not in html and "wave" not in html, name
        assert 'content="dark"' in html, name
