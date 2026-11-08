"""PEP 517 backend for an explicitly identified, minimal upstream wheel patch.

Builds execute this reviewed backend only. Upstream bytes are hash checked and
edited as archive data; their Python/build code is never imported or executed.
"""

import base64
import csv
import hashlib
import io
import json
import os
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent


def _spec():
    return json.loads((ROOT / "patch.json").read_text(encoding="utf-8"))


def _apply(path, content, spec):
    rules = spec.get("replacements", {}).get(path, [])
    if rules:
        text = content.decode("utf-8")
        for rule in rules:
            if text.count(rule["old"]) != 1:
                raise ValueError(f"Upstream patch context mismatch: {path}")
            text = text.replace(rule["old"], rule["new"], 1)
        return text.encode("utf-8")
    return content


def _metadata(spec):
    data = (ROOT / "METADATA").read_bytes()
    if hashlib.sha256(data).hexdigest() != spec["metadata_sha256"]:
        raise ValueError("Prepared metadata digest mismatch")
    return data


def get_requires_for_build_wheel(config_settings=None):
    del config_settings
    return []


def get_requires_for_build_sdist(config_settings=None):
    del config_settings
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    del config_settings
    spec = _spec()
    target = Path(metadata_directory) / spec["dist_info"]
    target.mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(_metadata(spec))
    wheel_metadata = (ROOT / "WHEEL").read_bytes()
    if hashlib.sha256(wheel_metadata).hexdigest() != spec["wheel_metadata_sha256"]:
        raise ValueError("Prepared WHEEL digest mismatch")
    (target / "WHEEL").write_bytes(wheel_metadata)
    upstream_license = ROOT / "UPSTREAM_LICENSE"
    if upstream_license.exists():
        (target / "licenses").mkdir(exist_ok=True)
        (target / "licenses" / "LICENSE").write_bytes(upstream_license.read_bytes())
    return spec["dist_info"]


def _upstream(spec):
    source = spec["upstream"]
    # A reviewed local wheelhouse permits entirely offline reproducibility.
    wheelhouse = os.environ.get("ESNYA_COMPAT_WHEELHOUSE")
    if wheelhouse:
        data = (Path(wheelhouse) / source["filename"]).read_bytes()
    else:
        if not source["url"].startswith("https://files.pythonhosted.org/packages/"):
            raise ValueError("Unexpected upstream artifact host")
        request = Request(
            source["url"], headers={"User-Agent": "esnya-dependency-compat-builder/1"}
        )
        with urlopen(request, timeout=90) as response:
            data = response.read(source["size"] + 1)
    if len(data) != source["size"] or hashlib.sha256(data).hexdigest() != source["sha256"]:
        raise ValueError("Upstream artifact size/digest mismatch")
    return data


def _check_prepared_file(name, data, spec):
    prefix = spec["upstream_dist_info"] + "/"
    if name == prefix + "METADATA":
        expected = _metadata(spec)
    elif name == prefix + "WHEEL":
        expected = (ROOT / "WHEEL").read_bytes()
    else:
        return
    if data != expected:
        raise ValueError("Built and prepared metadata differ")


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    del config_settings, metadata_directory
    spec = _spec()
    files = {}
    old_prefix = spec["upstream_dist_info"] + "/"
    with zipfile.ZipFile(io.BytesIO(_upstream(spec))) as archive:
        if sum(item.file_size for item in archive.infolist()) > 512 * 1024 * 1024:
            raise ValueError("Unexpected unpacked artifact size")
        for item in archive.infolist():
            name = item.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Unexpected archive path")
            if item.is_dir():
                continue
            if name == old_prefix + "RECORD":
                continue
            data = _apply(name, archive.read(item), spec)
            _check_prepared_file(name, data, spec)
            if name.startswith(old_prefix):
                name = spec["dist_info"] + "/" + name[len(old_prefix) :]
            if name in files:
                raise ValueError("Duplicate archive path")
            files[name] = data
    for patched in spec.get("replacements", {}):
        mapped = (
            spec["dist_info"] + "/" + patched[len(old_prefix) :]
            if patched.startswith(old_prefix)
            else patched
        )
        if mapped not in files:
            raise ValueError(f"Missing patched file: {patched}")
    record_name = spec["dist_info"] + "/RECORD"
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow([name, "sha256=" + digest, len(data)])
    writer.writerow([record_name, "", ""])
    files[record_name] = record.getvalue().encode()
    target = Path(wheel_directory)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        target / spec["wheel_filename"], "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data, compresslevel=9)
    return spec["wheel_filename"]


def build_sdist(sdist_directory, config_settings=None):
    del config_settings
    spec = _spec()
    name = spec["dist_info"].removesuffix(".dist-info")
    target = Path(sdist_directory)
    target.mkdir(parents=True, exist_ok=True)
    filename = name + ".tar.gz"
    with tarfile.open(target / filename, "w:gz") as archive:
        inputs = [
            "pyproject.toml",
            "build_backend.py",
            "patch.json",
            "METADATA",
            "WHEEL",
            "README.md",
        ]
        if (ROOT / "UPSTREAM_LICENSE").exists():
            inputs.append("UPSTREAM_LICENSE")
        for filename_in in inputs:
            source = ROOT / filename_in
            data = source.read_bytes()
            info = tarfile.TarInfo(name + "/" + filename_in)
            info.size = len(data)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(data))
    return filename
