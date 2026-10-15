# Uploading the prepared project

The repository root is the inner `LifeSyncAIPro` folder containing `manage.py`.
The project is published on branch `main` at
[Annarosejoshi/LifeSyncAIPro](https://github.com/Annarosejoshi/LifeSyncAIPro).
The `origin` remote is already configured; do not add it again for this checkout.
The complete project was merged with the earlier browser-upload history.

The latest files exclude `db.sqlite3`, and the local database was preserved
unchanged. A database copy from the earlier browser upload remains in the old
GitHub commits; removing a file from the current version does not remove it
from history. Historical removal would require a separately reviewed rewrite.

## Preparation verified locally

- Current Git-visible package: 398 files, approximately 25.96 MiB.
- A separate copy of only Git-visible files passed all 153 tests without the
  live user database, raw sources, or large historical model binaries.
- Independent artifact audit reproduced all four current models' metrics.
- Django configuration and migration checks passed.
- The configured VS Code virtual environment passed its dependency and Django checks.
- README local links and image references resolve. Four active model artifacts
  and their reduced evaluation datasets are included.
- No flagged files or common credential patterns were detected by the package
  check. This is not a comprehensive security audit.
- The optional legacy BRFSS page shows a current-screening link when its separate
  historical model is not installed; it is not required for the main dashboard.
- The latest runtime update passed all 157 application tests. Frontend assets
  are now bundled locally, and the Windows launcher performs readiness checks.
  The clean export also passed the 17 targeted asset/model/assessment tests.

## Review the upload first

```powershell
git status --short
python tools/check_repository.py
git add .
git diff --cached --stat
```

Review the staged files before committing. The live database, user uploads,
backups, environments, generated browser profiles, and large historical models
must remain excluded. Four active v10 artifacts and four reduced evaluation
datasets should be included. Do not force-add ignored personal or generated data.

## Commit and upload

Create an empty repository in your GitHub account. If this folder has no remote,
use its URL in place of the example below:

```powershell
git commit -m "Add LifeSync AI project"
git remote add origin https://github.com/YOUR-USERNAME/LifeSyncAIPro.git
git push -u origin main
```

If Git requests a name or email, configure your own Git identity, then retry.
If `origin` already exists, check `git remote -v` rather than adding it again.
Use your normal GitHub authentication; never paste tokens into README files,
source code, or a remote URL.

After uploading, open the GitHub **Actions** tab and review the Project checks
workflow. Local Windows checks passed before preparation; a remote Linux CI run
is a separate check and has not yet been performed.

Use Git rather than manually uploading the entire Windows project directory,
so the prepared ignore rules apply. Do not upload the outer folder containing
another nested copy of the project.

## New-computer check

Clone the repository into a new folder, follow README setup steps, run migrations,
register a new account, and run the tests. The repository intentionally contains
no account credentials or saved user database. SQLite is created locally when
migrations run. See `EVALUATION_GUIDE.md` for a demonstration sequence.
