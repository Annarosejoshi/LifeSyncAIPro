"""Check the Git-visible upload set without staging or printing secret values."""
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
result = subprocess.run(["git", "ls-files", "--cached", "--others",
                         "--exclude-standard", "-z"], cwd=ROOT,
                        capture_output=True, check=True)
names = sorted(set(result.stdout.decode().split("\0")) - {""})
issues = []
total = 0
secret = re.compile(rb"(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{36}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
for name in names:
    path = ROOT / name
    if not path.is_file():
        continue
    size = path.stat().st_size
    total += size
    if size > 50 * 1024**2:
        issues.append(f"Large file requires review: {name}")
    parts = Path(name).parts
    if any(part in {"venv", "venv_old", ".venv", "media", "backups", "artifacts", "__pycache__"} for part in parts):
        issues.append(f"Local/private directory included: {name}")
    if path.suffix in {".sqlite3", ".db", ".log", ".pem", ".key"} or path.name == ".env":
        issues.append(f"Local/private file included: {name}")
    if path.suffix == ".json":
        try:
            json.loads(path.read_text(encoding="utf-8-sig"))
        except (ValueError, UnicodeError) as exc:
            issues.append(f"Invalid JSON: {name}: {type(exc).__name__}")
    if size < 5 * 1024**2 and path.suffix in {".py", ".md", ".json", ".txt", ".html", ".yml", ".yaml", ".bat"}:
        if secret.search(path.read_bytes()):
            issues.append(f"Potential credential found; inspect locally: {name}")

for model in ("diabetes", "hypertension", "cardiovascular", "mental_health"):
    for required in (f"ml/models/{model}_lifestyle_v10.joblib",
                     f"ml/models/{model}_lifestyle_v10_metadata.json",
                     f"ml/data/processed/{model}_lifestyle_v2.csv"):
        if required not in names:
            issues.append(f"Required model/evaluation file excluded: {required}")

print(f"Upload candidates: {len(names)} files, {total / 1024**2:.2f} MiB")
if issues:
    print("\n".join(issues))
    sys.exit(1)
print("PASS: required active artifacts included; no flagged files, invalid JSON, or detected credential patterns.")
print("This pattern check is not a comprehensive secret scan. Review the staged changes before publishing.")
