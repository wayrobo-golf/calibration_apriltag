"""Canonical JSON, atomic writes, and raw-data identity manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    payload = canonical_json_bytes(value) + b"\n"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_raw_data_identity(root: Path) -> dict[str, Any]:
    files = []
    if root.exists():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            files.append(
                {
                    "path": relative,
                    "size_bytes": stat.st_size,
                    "sha256": sha256_file(path),
                }
            )
    content = {"schema_version": 1, "root_name": root.name, "files": files}
    content["identity_sha256"] = sha256_bytes(canonical_json_bytes(content))
    return content


def write_checksums(root: Path, *, excluded: tuple[str, ...] = ("CHECKSUMS.sha256",)) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        lines.append(f"{sha256_file(path)}  {relative}\n")
    destination = root / "CHECKSUMS.sha256"
    temporary = root / ".CHECKSUMS.sha256.incomplete"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.writelines(lines)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
