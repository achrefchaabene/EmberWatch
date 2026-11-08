"""Download the datasets listed in configs/datasets.yaml.

One function per source type, all with the same contract:
    fetch_<source>(name, spec, dest: Path) -> Path   (folder that contains the data)

Design rules:
- Idempotent: a dataset already present (with a .complete marker) is not downloaded again.
  Re-running a Kaggle cell costs nothing.
- Safe extraction: archive members cannot write outside the target folder (zip-slip).
- No secret in code or config: API keys come from env vars or Kaggle Secrets.
- A manifest (data/raw/manifest.json) records where each dataset is, so the next
  steps (EDA, prepare_dataset) never guess paths.

Usage:
    python -m src.detection.download --only dfire hpwren_smoke --out data/raw
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"
MARKER = ".complete"  # written when a dataset finished downloading + extracting


# ------------------------------------------------------------------ helpers
def load_registry(path: Path = DEFAULT_CONFIG) -> dict:
    """Read the dataset registry (name -> spec)."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["datasets"]


def get_secret(name: str) -> str | None:
    """Read a secret from the environment, then from Kaggle Secrets when on Kaggle."""
    if os.environ.get(name):
        return os.environ[name]
    try:  # only exists inside a Kaggle notebook
        from kaggle_secrets import UserSecretsClient

        return UserSecretsClient().get_secret(name)
    except Exception:  # noqa: BLE001 - not on Kaggle, or secret not attached
        return None


def _is_within(base: Path, target: Path) -> bool:
    """True if target stays inside base once resolved (blocks '../' tricks)."""
    base, target = base.resolve(), target.resolve()
    return base == target or base in target.parents


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract zip / tar(.gz) archives, refusing any member that escapes dest."""
    dest.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not _is_within(dest, dest / member):
                    raise ValueError(f"Unsafe path in archive: {member}")
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                if member.issym() or member.islnk() or not _is_within(dest, dest / member.name):
                    raise ValueError(f"Unsafe member in archive: {member.name}")
            try:
                tf.extractall(dest, filter="data")  # official safe filter (Python >= 3.11.4 / 3.12)
            except TypeError:
                tf.extractall(dest)  # older Python: our own checks above already ran
    else:
        raise ValueError(f"Not a zip or tar archive: {archive}")


def http_download(url: str, out_file: Path, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk (no full file in RAM), via a .part file to survive interruptions."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(out_file.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "EmberWatch-dataset-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as fh:
        while block := response.read(chunk):
            fh.write(block)
    tmp.replace(out_file)  # atomic: a half-downloaded file never looks complete
    return out_file


def onedrive_direct_url(share_url: str) -> str:
    """Convert a OneDrive *file* share link to a direct-download URL (OneDrive shares API)."""
    token = base64.urlsafe_b64encode(share_url.encode()).decode().rstrip("=")
    return f"https://api.onedrive.com/v1.0/shares/u!{token}/root/content"


def _done(dest: Path) -> bool:
    return (dest / MARKER).exists()


def _mark_done(dest: Path, spec: dict) -> None:
    (dest / MARKER).write_text(json.dumps({"source": spec.get("source"),
                                           "at": datetime.now(timezone.utc).isoformat()}))


# ------------------------------------------------------------------ sources
def fetch_kaggle(name: str, spec: dict, dest: Path) -> Path:
    """Kaggle dataset: use the attached copy on Kaggle, else download with kagglehub."""
    attached = spec.get("kaggle_input")
    if attached and Path(attached).exists():
        return Path(attached)  # read-only input, nothing to copy
    import kagglehub  # credentials: ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY

    return Path(kagglehub.dataset_download(spec["handle"], output_dir=str(dest)))


def fetch_roboflow(name: str, spec: dict, dest: Path) -> Path:
    """Roboflow export (YOLO format) through the official SDK."""
    key = get_secret("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY missing: set it as an env var or a Kaggle Secret.")
    from roboflow import Roboflow

    project = Roboflow(api_key=key).workspace(spec["workspace"]).project(spec["project"])
    project.version(int(spec["version"])).download(spec.get("format", "yolov8"), location=str(dest), overwrite=True)
    return dest


def fetch_http(name: str, spec: dict, dest: Path) -> Path:
    """One or several direct URLs (base_url + files, or url). Archives are extracted."""
    urls = [spec["url"]] if spec.get("url") else [spec["base_url"].rstrip("/") + "/" + f for f in spec.get("files", [])]
    if not urls:
        raise ValueError(f"{name}: no url / files listed in configs/datasets.yaml")
    for url in urls:
        archive = http_download(url, dest / "_downloads" / url.rsplit("/", 1)[-1])
        if spec.get("extract", True) and (zipfile.is_zipfile(archive) or tarfile.is_tarfile(archive)):
            safe_extract(archive, dest)
            archive.unlink()  # free disk space (Kaggle /kaggle/working is limited)
    shutil.rmtree(dest / "_downloads", ignore_errors=True)
    return dest


def fetch_gdrive(name: str, spec: dict, dest: Path) -> Path:
    """Google Drive file or folder link (must be shared as 'anyone with the link')."""
    import gdown

    dest.mkdir(parents=True, exist_ok=True)
    if "/folders/" in spec["url"]:
        gdown.download_folder(url=spec["url"], output=str(dest), quiet=False)
        return dest
    out = Path(gdown.download(url=spec["url"], output=str(dest) + os.sep, quiet=False, fuzzy=True))
    if spec.get("extract", True) and (zipfile.is_zipfile(out) or tarfile.is_tarfile(out)):
        safe_extract(out, dest)
        out.unlink()
    return dest


def fetch_github(name: str, spec: dict, dest: Path) -> Path:
    """Shallow clone of a repository (code, small data, download scripts)."""
    if dest.exists() and any(dest.iterdir()):
        return dest
    subprocess.run(["git", "clone", "--depth", "1", spec["url"], str(dest)], check=True)
    return dest


def fetch_onedrive(name: str, spec: dict, dest: Path) -> Path:
    """OneDrive *file* link -> direct download. Folder links cannot be fetched this way."""
    if "/f/" in spec["url"]:
        raise ValueError(f"{name}: OneDrive folder links are not downloadable by script; "
                         "download it in a browser or use the Kaggle source.")
    return fetch_http(name, {"url": onedrive_direct_url(spec["url"]), "extract": spec.get("extract", True)}, dest)


FETCHERS = {"kaggle": fetch_kaggle, "roboflow": fetch_roboflow, "http": fetch_http,
            "gdrive": fetch_gdrive, "github": fetch_github, "onedrive": fetch_onedrive}


# ------------------------------------------------------------------ entry point
def download(names: list[str], out: Path, config: Path = DEFAULT_CONFIG) -> dict[str, str]:
    """Fetch every requested dataset; return and save {name: local path}."""
    registry = load_registry(config)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in names:
        if name not in registry:
            raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(registry)}")
        spec, dest = registry[name], out / name
        if _done(dest):
            print(f"[skip] {name}: already in {dest}")
            manifest[name] = str(dest)
            continue
        print(f"[get ] {name} ({spec['source']})")
        path = FETCHERS[spec["source"]](name, spec, dest)
        if path == dest:
            _mark_done(dest, spec)
        manifest[name] = str(path)
        print(f"[ok  ] {name} -> {path}")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", required=True, help="dataset names from configs/datasets.yaml")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(download(args.only, args.out, args.config), indent=2))


if __name__ == "__main__":
    main()
