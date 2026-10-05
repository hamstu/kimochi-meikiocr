#!/usr/bin/env python3
"""Download pinned upstream models. Requires Python 3.11+, no third-party packages."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

MODELS = [
    ("rtr46/meiki.text.detect.v0", "a9cffa4f60cbf72ddb87edf19c6f98a01cd042e6", "meiki.text.detect.v0.1.960x544.onnx"),
    ("rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375", "meiki.text.rec.v0.960x32.onnx"),
    ("rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375", "meiki.text.rec.v0.vertical.32x480.onnx"),
]

def download(repo, revision, name, directory):
    url = f"https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url) as response:
        metadata = json.load(response)
    entry = next(s for s in metadata["siblings"] if s["rfilename"] == name)
    expected = entry["lfs"]["sha256"]
    target = directory / name
    if target.exists():
        with target.open("rb") as existing:
            actual = hashlib.file_digest(existing, "sha256").hexdigest()
        if actual == expected:
            print(f"Verified {target}")
            return
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            with urllib.request.urlopen(url) as response:
                while block := response.read(1024 * 1024):
                    output.write(block)
                    digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"SHA-256 mismatch for {name}")
        os.replace(temporary, target)
        print(f"Downloaded and verified {target}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path("models"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        download(*model, args.directory)
