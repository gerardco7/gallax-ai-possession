"""Content-addressed experiment records, feature caches and atomic artifact writes.

No match arrays are written to manifests. Paths identify inputs; hashes identify their bytes.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_HASHES = {}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=lambda x: x.item() if isinstance(x, np.generic) else str(x))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASHES:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        _HASHES[key] = h.hexdigest()
    return _HASHES[key]


def inputs(paths):
    return {str(Path(p)): file_hash(p) for p in sorted(map(str, paths))}


def source_snapshot():
    sources = sorted((ROOT / "possession").glob("*.py")) + sorted((ROOT / "evaluation").glob("*.py"))
    sources += sorted((ROOT / "configs").glob("*.json"))
    sources += [ROOT / "games.py"] + sorted(ROOT.glob("requirements*.txt"))
    hashes = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "files": hashes, "sha256": digest(hashes)}


def runtime():
    versions = {}
    for package in ("numpy", "pandas", "torch", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def atomic_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix="." + path.name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def cached_features(path, input_paths, schema, build):
    """Old/unversioned, stale or incomplete caches are rebuilt, never silently reused."""
    key = {"schema": schema, "inputs": inputs(input_paths)}
    path = Path(path)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if str(z["cache_key"].item()) == canonical(key):
                    return z["feats"], z["centres"]
        except (KeyError, ValueError, OSError, EOFError):
            pass
    feats, centres = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temp, feats=feats, centres=centres, cache_key=canonical(key))
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return feats, centres


class Run:
    """Declare before scoring; resume only the identical source/config/input/runtime identity.

    Outputs live under runs/<name>/<id>, so a different experiment cannot overwrite them.
    Completed runs reject changed results. Old flat CSVs remain historical, not resumable state.
    """
    def __init__(self, root, name, config, splits, input_paths=(), protocol=None):
        self.identity = {"format": 1, "name": name, "config": config, "splits": splits,
                         "inputs": inputs(input_paths), "source": source_snapshot(),
                         "runtime": runtime(), "protocol": protocol or {}}
        self.id = digest(self.identity)
        self.path = Path(root) / "runs" / name / self.id[:20]
        self.path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.path / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            if "results_sha256" in self.manifest:
                result = self.path / "results.csv"
                if not result.exists() or file_hash(result) != self.manifest["results_sha256"]:
                    raise ValueError("Recorded run results missing or changed")
                for relative, expected in self.manifest.get("artifacts", {}).items():
                    artifact = self.path / relative
                    if not artifact.exists() or file_hash(artifact) != expected:
                        raise ValueError("Recorded run artifact missing or changed: " + relative)
            if self.manifest["identity"] != self.identity:
                raise ValueError("Run identity mismatch")
        else:
            self.manifest = {"id": self.id, "identity": self.identity, "status": "declared",
                             "created_utc": datetime.now(timezone.utc).isoformat()}
            for relative, expected in self.identity["source"]["files"].items():
                content = (ROOT / relative).read_bytes()
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("Source changed while declaring the run")
                atomic_bytes(self.path / "source" / relative, content)
            write_json(self.manifest_path, self.manifest)

    def results(self, frame, complete=False):
        content = frame.to_csv(index=False).encode()
        path = self.path / "results.csv"
        if self.manifest["status"] == "complete":
            if path.read_bytes() != content:
                raise ValueError("Completed run results are immutable; use a new experiment identity")
            return path
        atomic_bytes(path, content)
        self.manifest.update(status="complete" if complete else "running", results_sha256=file_hash(path))
        if complete:
            self.manifest["artifacts"] = {
                str(p.relative_to(self.path)): file_hash(p) for p in sorted(self.path.rglob("*"))
                if p.is_file() and p != self.manifest_path
                and p.relative_to(self.path).parts[0] not in ("source", "baseline")}

        write_json(self.manifest_path, self.manifest)
        return path
