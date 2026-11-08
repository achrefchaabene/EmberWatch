"""Tests for the dataset downloader (no internet: a local HTTP server stands in for HPWREN)."""

import functools
import http.server
import io
import tarfile
import threading
import zipfile

import pytest

from src.detection import download as dl


def test_registry_lists_m2_datasets():
    reg = dl.load_registry()
    assert reg["dfire"]["source"] == "kaggle"
    assert reg["hpwren_smoke"]["source"] == "roboflow"
    assert all(spec["source"] in dl.FETCHERS for spec in reg.values())


def test_onedrive_share_link_becomes_direct_api_url():
    url = dl.onedrive_direct_url("https://1drv.ms/u/s!ABC")
    assert url.startswith("https://api.onedrive.com/v1.0/shares/u!") and url.endswith("/root/content")
    assert "=" not in url.split("u!")[1]  # base64url without padding


def test_onedrive_folder_link_is_refused(tmp_path):
    with pytest.raises(ValueError, match="folder"):
        dl.fetch_onedrive("x", {"url": "https://1drv.ms/f/c/abc"}, tmp_path)


def test_safe_extract_blocks_zip_slip(tmp_path):
    # An archive member trying to write outside the target folder must be refused.
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as zf:
        zf.writestr("../outside.txt", "pwned")
    with pytest.raises(ValueError, match="Unsafe"):
        dl.safe_extract(evil, tmp_path / "dest")
    assert not (tmp_path / "outside.txt").exists()


def test_roboflow_without_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ROBOFLOW_API_KEY"):
        dl.fetch_roboflow("hpwren_smoke", {"workspace": "w", "project": "p", "version": 1}, tmp_path)


def test_kaggle_uses_attached_input_without_downloading(tmp_path):
    attached = tmp_path / "kaggle_input"
    attached.mkdir()
    assert dl.fetch_kaggle("dfire", {"kaggle_input": str(attached), "handle": "x/y"}, tmp_path / "d") == attached


@pytest.fixture()
def http_server(tmp_path):
    """Serve tmp_path/www over real HTTP on localhost (stands in for cdn.hpwren.ucsd.edu)."""
    www = tmp_path / "www"
    www.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(www))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield www, f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def test_http_tgz_download_extract_manifest_and_skip(tmp_path, http_server):
    www, base = http_server
    # Build a FIgLib-like sequence archive: <sequence>/<offset>.jpg
    with tarfile.open(www / "20190529_94Fire_lp-s-mobo-c.tgz", "w:gz") as tf:
        data = b"fake-jpeg"
        info = tarfile.TarInfo("20190529_94Fire_lp-s-mobo-c/1559149680_+00000.jpg")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    cfg = tmp_path / "datasets.yaml"
    cfg.write_text(f"datasets:\n  figlib_sample:\n    source: http\n    base_url: {base}\n"
                   f"    files: [20190529_94Fire_lp-s-mobo-c.tgz]\n    extract: true\n")
    out = tmp_path / "raw"

    manifest = dl.download(["figlib_sample"], out, cfg)

    frame = out / "figlib_sample" / "20190529_94Fire_lp-s-mobo-c" / "1559149680_+00000.jpg"
    assert frame.read_bytes() == b"fake-jpeg"
    assert manifest["figlib_sample"] == str(out / "figlib_sample")
    assert not (out / "figlib_sample" / "_downloads").exists()  # archive removed to save disk
    # Second call must not download again (idempotent Kaggle cells).
    (www / "20190529_94Fire_lp-s-mobo-c.tgz").unlink()
    assert dl.download(["figlib_sample"], out, cfg)["figlib_sample"] == str(out / "figlib_sample")


def test_unknown_dataset_name(tmp_path):
    with pytest.raises(KeyError, match="Unknown dataset"):
        dl.download(["does_not_exist"], tmp_path)


def test_kaggle_finds_new_and_old_mount_layouts(tmp_path, monkeypatch):
    # New Kaggle layout: /kaggle/input/datasets/<owner>/<slug> (seen on Kaggle, Nov 2026)
    spec = {"handle": "sayedgamal99/smoke-fire-detection-yolo"}
    cands = dl.kaggle_mount_candidates(spec, root=tmp_path)
    assert cands == [tmp_path / "datasets" / "sayedgamal99" / "smoke-fire-detection-yolo",
                     tmp_path / "smoke-fire-detection-yolo"]
    new = cands[0]
    new.mkdir(parents=True)
    monkeypatch.setattr(dl, "KAGGLE_INPUT", tmp_path)
    assert dl.fetch_kaggle("dfire", spec, tmp_path / "dl") == new  # no download attempted
