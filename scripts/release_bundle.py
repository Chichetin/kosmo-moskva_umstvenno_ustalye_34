"""Release bundle for GitVerse (#85): run kit + weights as two deterministic .zip + SHA256SUMS.

outputs/ and *.tif are not in git, so the 12 required files of the demo chip reach the jury
only as a release asset. Build (default, no network):
    uv run --frozen python scripts/release_bundle.py --out outputs/<run_id>
    make release OUT=outputs/<run_id>
Then check: cd <dest> && sha256sum -c SHA256SUMS
Upload to a GitVerse release (REST API, docs: gitverse.ru/docs/developers/public-api/releases):
    make release OUT=outputs/<run_id> TAG=<tag>

Refuses when `python -m floodvalue validate --out <run>` is not 12/12, when models/ holds a Git
LFS pointer instead of content, when models/ has no weight files, or when a weight listed in
run_metadata.json is missing or has another sha256, or when the repo's models/ differs from
models/ at the kit's commit (edits or untracked files). Upload refuses to overwrite assets, to
reuse a tag that points to another commit, and to publish the reference fixture. Archives are byte-identical on rebuild:
sorted entries, fixed timestamp and permissions. The token (GITVERSE_TOKEN from the environment
or .env) goes only into the Authorization header — never to stdout, argv, URL or files.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    "flood_probability.tif", "flood_mask.tif", "assets.geojson", "assets.csv", "asset_loss.csv",
    "candidate_orders.geojson", "procurement_plan.csv", "strategy_plans.json",
    "strategy_comparison.csv", "sensitivity.csv", "run_metadata.json", "source_manifest.json",
)
# Weight extensions tracked by Git LFS (.gitattributes).
WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".joblib", ".pkl", ".safetensors", ".ckpt", ".npz"}
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
MAX_ASSET_BYTES = 100 * 1024 * 1024  # GitVerse: 100 MB per release file
MAX_ASSETS_PER_UPLOAD = 5            # GitVerse: 5 files per upload
DEFAULT_API = "https://api.gitverse.ru/repos/hackrus.experts/kosmo-moskva_umstvenno_ustalye_34"
ACCEPT = "application/vnd.gitverse.object+json;version=1"
SUMS = "SHA256SUMS"


class ReleaseError(Exception):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files_under(root: Path) -> list[Path]:
    """Regular files under root, sorted by POSIX relative path."""
    return sorted((p for p in root.rglob("*") if p.is_file()),
                  key=lambda p: p.relative_to(root).as_posix())


def write_zip(dest: Path, root: Path, prefix: str) -> None:
    """Deterministic zip: sorted entries, fixed mtime/permissions/system, deflate level 9."""
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w") as z:
        for path in files_under(root):
            info = zipfile.ZipInfo(f"{prefix}/{path.relative_to(root).as_posix()}", ZIP_EPOCH)
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes(), compresslevel=9)
    tmp.replace(dest)


def check_run(run_dir: Path) -> dict:
    """The validator must report 12/12; returns run_metadata.json."""
    if not run_dir.is_dir():
        raise ReleaseError(f"каталог комплекта не найден: {run_dir}")
    res = subprocess.run([sys.executable, "-m", "floodvalue", "validate", "--out", str(run_dir)],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    print(res.stdout, end="")
    if res.returncode != 0 or "12/12" not in res.stdout:
        raise ReleaseError(f"валидатор не дал 12/12 для {run_dir} (код {res.returncode}); "
                           f"комплект не упакован\n{res.stderr}")
    missing = [n for n in REQUIRED_FILES if not (run_dir / n).is_file()]
    if missing:
        raise ReleaseError(f"нет обязательных файлов: {', '.join(missing)}")
    return json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))


def check_models(models_dir: Path) -> None:
    """Every file is content, not an LFS pointer; at least one weight file."""
    if not models_dir.is_dir():
        raise ReleaseError(f"каталог весов не найден: {models_dir}")
    files = files_under(models_dir)
    pointers = []
    for path in files:
        with path.open("rb") as f:
            if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                pointers.append(path.relative_to(models_dir).as_posix())
    if pointers:
        raise ReleaseError("в models/ LFS-указатель вместо содержимого (сделайте `git lfs pull`): "
                           + ", ".join(pointers))
    if not any(p.suffix in WEIGHT_SUFFIXES for p in files):
        raise ReleaseError(f"в {models_dir} нет файлов весов ({', '.join(sorted(WEIGHT_SUFFIXES))})")


def check_weights_match(metadata: dict, models_dir: Path) -> None:
    """Weights named in run_metadata.methods.*.weights (paths models/...) = archived files."""
    for method, info in sorted((metadata.get("methods") or {}).items()):
        if not isinstance(info, dict):
            continue
        for w in info.get("weights") or []:
            rel = str(w.get("path", ""))
            if not rel.startswith("models/"):
                continue
            path = models_dir / rel.removeprefix("models/")
            if not path.is_file():
                raise ReleaseError(f"{method}: вес {rel} из run_metadata.json отсутствует в {models_dir}")
            if sha256_file(path) != w.get("sha256"):
                raise ReleaseError(f"{method}: sha256 {rel} не совпадает с run_metadata.json — "
                                   "веса не те, что дали комплект")


def check_models_committed(repo: Path, models_dir: Path, commit: str) -> None:
    """models/ in the working tree = models/ at the kit's commit: no edits, no untracked files."""
    res = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                         capture_output=True, text=True)
    if res.returncode != 0:
        raise ReleaseError(f"commit {commit} из run_metadata.json не найден в git")
    rel = models_dir.resolve().relative_to(repo.resolve()).as_posix()
    changed = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", res.stdout.strip(), "--", rel],
                             capture_output=True, text=True, check=True).stdout.split()
    untracked = subprocess.run(["git", "-C", str(repo), "ls-files", "--others", "--", rel], capture_output=True, text=True, check=True).stdout.split()
    if changed or untracked:
        raise ReleaseError(f"{rel}/ отличается от commit {commit}, которым собран комплект "
                           f"(изменены: {', '.join(changed) or '—'}; не в git: {', '.join(untracked) or '—'})")


def is_reference(run_dir: Path, metadata: dict) -> bool:
    return (run_dir.resolve() == (REPO_ROOT / "tests" / "fixtures" / "good").resolve()
            or metadata.get("commit") == "fixture")


def build(run_dir: Path, models_dir: Path, dest: Path) -> list[Path]:
    """All checks first, then two zips and SHA256SUMS in dest; returns the zip paths."""
    metadata = check_run(run_dir)
    check_models(models_dir)
    check_weights_match(metadata, models_dir)
    commit = str(metadata.get("commit") or "unknown")
    if not is_reference(run_dir, metadata) and models_dir.resolve() == (REPO_ROOT / "models").resolve():
        check_models_committed(REPO_ROOT, models_dir, commit)
    run_id = str(metadata.get("run_id") or run_dir.name)
    dest.mkdir(parents=True, exist_ok=True)
    run_zip = dest / f"run_{run_id}.zip"
    models_zip = dest / f"models_{commit}.zip"
    write_zip(run_zip, run_dir, run_id)
    write_zip(models_zip, models_dir, "models")
    zips = [run_zip, models_zip]
    lines = sorted(f"{sha256_file(p)}  {p.name}" for p in zips)
    (dest / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return zips


# --- upload ------------------------------------------------------------------------------------

def _token() -> str:
    token = os.environ.get("GITVERSE_TOKEN", "")
    env_file = REPO_ROOT / ".env"
    if not token and env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "GITVERSE_TOKEN":
                token = value.strip().strip("'\"")
    if not token or token == "your_token_here":
        raise ReleaseError("GITVERSE_TOKEN не задан (окружение или .env)")
    return token


def _request(method: str, url: str, token: str, data: bytes | None = None,
             content_type: str | None = None) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}", "Accept": ACCEPT}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    try:
        body = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        body = {"raw": raw[:300].decode("utf-8", "replace")}
    return status, body


def _multipart(path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"attachment\"; "
            f"filename=\"{path.name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
    return head + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode(), \
        f"multipart/form-data; boundary={boundary}"


def upload(tag: str, files: list[Path], *, target: str, body: str) -> None:
    """Release by tag (created if absent), then each file as an asset; no overwrite."""
    if len(files) > MAX_ASSETS_PER_UPLOAD:
        raise ReleaseError(f"GitVerse принимает не больше {MAX_ASSETS_PER_UPLOAD} файлов за загрузку")
    for p in files:
        if p.stat().st_size > MAX_ASSET_BYTES:
            raise ReleaseError(f"{p.name}: {p.stat().st_size} байт — больше лимита GitVerse 100 МБ на файл")
    token = _token()
    api = os.environ.get("GITVERSE_API", DEFAULT_API).rstrip("/")
    status, release = _request("GET", f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token)
    if status == 404:
        payload = json.dumps({"tag_name": tag, "name": tag, "target_commitish": target, "body": body,
                              "draft": False, "prerelease": False, "is_authorized_only": False})
        status, release = _request("POST", f"{api}/releases", token, payload.encode(), "application/json")
        if status != 201:
            raise ReleaseError(f"релиз {tag} не создан: HTTP {status} {release}")
        print(f"[release] создан релиз {tag} (id {release.get('id')}) на {target}")
    elif status != 200:
        raise ReleaseError(f"релиз {tag}: HTTP {status} {release}")
    else:
        existing_target = str(release.get("target_commitish") or "")
        if existing_target and not target.startswith(existing_target):
            raise ReleaseError(f"релиз {tag} уже есть и указывает на {existing_target}, а комплект "
                               f"собран на {target} — возьмите новый тег")
    existing = {a.get("name") for a in release.get("assets") or []}
    clash = sorted(existing & {p.name for p in files})
    if clash:
        raise ReleaseError(f"в релизе {tag} уже есть {', '.join(clash)} — удалите их в GitVerse "
                           "или возьмите новый тег; перезапись не делается")
    for p in files:
        data, ctype = _multipart(p)
        url = f"{api}/releases/{release['id']}/assets?name={urllib.parse.quote(p.name)}"
        status, asset = _request("POST", url, token, data, ctype)
        if status != 201:
            raise ReleaseError(f"{p.name}: загрузка не удалась, HTTP {status} {asset}")
        print(f"[release] загружен {p.name} ({p.stat().st_size} байт)")


def _resolve_commit(commit: str) -> str:
    res = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                         capture_output=True, text=True, cwd=REPO_ROOT)
    if res.returncode != 0:
        raise ReleaseError(f"commit {commit} из run_metadata.json не найден в git")
    return res.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Архивы комплекта и весов для релиза GitVerse (#85)")
    parser.add_argument("--out", required=True, type=Path, help="каталог комплекта outputs/<run_id>")
    parser.add_argument("--models", type=Path, default=REPO_ROOT / "models", help="каталог весов")
    parser.add_argument("--dest", type=Path, default=None,
                        help="куда положить архивы (по умолчанию $FLOODVALUE_OUTPUT_DIR/release)")
    parser.add_argument("--upload", metavar="TAG", default=None,
                        help="загрузить архивы и SHA256SUMS в релиз GitVerse с этим тегом")
    args = parser.parse_args(argv)
    dest = args.dest or Path(os.environ.get("FLOODVALUE_OUTPUT_DIR", REPO_ROOT / "outputs")) / "release"
    try:
        zips = build(args.out, args.models, dest)
        metadata = json.loads((args.out / "run_metadata.json").read_text(encoding="utf-8"))
        if is_reference(args.out, metadata):
            print("[release] ЭТАЛОН tests/fixtures/good — проверка инструмента, не комплект для релиза")
        for line in (dest / SUMS).read_text(encoding="utf-8").splitlines():
            print(f"[release] {line}")
        for p in zips:
            size = p.stat().st_size
            flag = "  ВНИМАНИЕ: больше 100 МБ, в релиз GitVerse не пройдёт" if size > MAX_ASSET_BYTES else ""
            print(f"[release] {p} — {size} байт{flag}")
        print(f"[release] проверка: cd {dest} && sha256sum -c {SUMS}")
        if args.upload:
            if is_reference(args.out, metadata):
                raise ReleaseError("эталон tests/fixtures/good в релиз не загружается")
            if metadata.get("commit_dirty"):
                raise ReleaseError("комплект собран из рабочего дерева с незакоммиченными правками "
                                   "(run_metadata.commit_dirty=true) — commit не воспроизводит его")
            target = _resolve_commit(str(metadata.get("commit", "")))
            body = (f"Комплект запуска {metadata.get('run_id')} (commit {target}) и веса models/.\n"
                    f"Проверка: sha256sum -c {SUMS}\n\n" + (dest / SUMS).read_text(encoding="utf-8"))
            upload(args.upload, [*zips, dest / SUMS], target=target, body=body)
    except ReleaseError as e:
        print(f"[release] ОТКАЗ: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
