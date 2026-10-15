"""Download pinned frontend assets once; the application serves them locally.

Run only when intentionally refreshing vendored files: python tools/vendor_assets.py
"""
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1] / "static/vendor"
FILES = {
    "bootstrap/bootstrap.min.css": "bootstrap@5.3.8/dist/css/bootstrap.min.css",
    "bootstrap/bootstrap.bundle.min.js": "bootstrap@5.3.8/dist/js/bootstrap.bundle.min.js",
    "bootstrap/LICENSE": "bootstrap@5.3.8/LICENSE",
    "bootstrap-icons/bootstrap-icons.css": "bootstrap-icons@1.13.1/font/bootstrap-icons.css",
    "bootstrap-icons/fonts/bootstrap-icons.woff2": "bootstrap-icons@1.13.1/font/fonts/bootstrap-icons.woff2",
    "bootstrap-icons/fonts/bootstrap-icons.woff": "bootstrap-icons@1.13.1/font/fonts/bootstrap-icons.woff",
    "bootstrap-icons/LICENSE": "bootstrap-icons@1.13.1/LICENSE",
    "chartjs/chart.umd.js": "chart.js@4.5.1/dist/chart.umd.js",
    "chartjs/LICENSE.md": "chart.js@4.5.1/LICENSE.md",
}
BOOTSTRAP_SRI = {
    "bootstrap/bootstrap.min.css": "sRIl4kxILFvY47J16cr9ZwB07vP4J8+LH7qKQnuqkuIAvNWLzeN8tE5YBujZqJLB",
    "bootstrap/bootstrap.bundle.min.js": "FKyoEForCGlyvwx9Hj09JcYn3nv7wiPVlz7YYwJrWVcXK/BmnVDxM+D2scQbITxI",
}


def download(item):
    name, package = item
    error = None
    for host in ("https://cdn.jsdelivr.net/npm/", "https://unpkg.com/"):
        url = host + package
        try:
            with urlopen(Request(url, headers={"User-Agent": "LifeSync-asset-preparation"}), timeout=20) as response:
                content = response.read()
            if b"<html" in content[:300].lower() or len(content) < 20:
                raise ValueError(f"Unexpected asset response: {name}")
            if name in BOOTSTRAP_SRI:
                assert base64.b64encode(hashlib.sha384(content).digest()).decode() == BOOTSTRAP_SRI[name], name
            target = ROOT / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            print(f"Saved {name} ({len(content)} bytes)", flush=True)
            return name, dict(source=url, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        except Exception as exc:
            error = exc
    raise RuntimeError(f"Could not download {name}: {error}")


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as executor:
        manifest = dict(executor.map(download, FILES.items()))
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
