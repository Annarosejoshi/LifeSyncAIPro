"""Render synthetic-user previews without keeping any database changes.

Run with working Python: python tools/review_ui.py
Screenshots use an installed Chrome; these are visual checks, not a live E2E test.
"""
import os
import base64
import mimetypes
from pathlib import Path
import re
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
import django
django.setup()

from django.contrib.auth.models import User
from django.db import transaction
from django.test import Client, override_settings
from django.utils import timezone
from accounts.models import UserProfile
from lifestyle.models import DailyCheckIn
from mental_health.models import MentalWellnessAssessment

OUT = ROOT / "artifacts/ui-review"
OUT.mkdir(parents=True, exist_ok=True)


def inline_assets(html):
    def stylesheet(match):
        path = ROOT / "static" / match[1]
        if not path.exists():
            return match[0]
        def local_url(asset):
            url = asset[1].strip("\"'")
            if url.startswith("data:"):
                return asset[0]
            local = path.parent / url.split("?")[0]
            if local.is_file():
                mime = mimetypes.guess_type(str(local))[0] or "application/octet-stream"
                return f'url("data:{mime};base64,{base64.b64encode(local.read_bytes()).decode()}")'
            return asset[0]
        css = re.sub(r'url\(([^)]+)\)', local_url, path.read_text(encoding="utf-8-sig"))
        return "<style>" + css + "</style>"
    html = re.sub(r'<link[^>]*href="/static/([^"?]+)(?:\?[^" ]*)?"[^>]*>', stylesheet, html)
    def javascript(match):
        path = ROOT / "static" / match[1]
        return "<script>" + path.read_text(encoding="utf-8-sig") + "</script>" if path.exists() else match[0]
    html = re.sub(r'<script[^>]*src="/static/([^"?]+)(?:\?[^" ]*)?"[^>]*></script>', javascript, html)
    html = html.replace('<head>', '<head><script>window.reviewErrors=[];window.addEventListener("error",event=>{if(event.message)reviewErrors.push(event.message)});</script>', 1)
    return html.replace('</body>', "<script>document.body.dataset.reviewWidth=JSON.stringify({viewport:innerWidth,scroll:document.documentElement.scrollWidth});</script></body>")


with override_settings(ALLOWED_HOSTS=["testserver"]), transaction.atomic():
    user = User.objects.create_user(username="ui_review_" + uuid4().hex[:10], first_name="Demo")
    UserProfile.objects.filter(user=user).update(
        date_of_birth="1990-01-01", gender="female", height_cm=165,
        weight_kg=64, smoking_status="never")
    DailyCheckIn.objects.create(user=user, date=timezone.localdate(),
        sleep_hours=7, water_liters=2, exercise_minutes=30, screen_time_hours=4,
        mood=4, stress_level=2, energy_level=4)
    MentalWellnessAssessment.objects.create(user=user, date=timezone.localdate(),
        mood=4, stress=2, anxiety=2, energy=4, motivation=4, concentration=4,
        social_wellbeing=4, workload=2)
    client = Client()
    client.force_login(user)
    for name, url in (("dashboard", "/"), ("predictions", "/risks/"),
                      ("checkin", "/check-in/"), ("wellness", "/reports/"),
                      ("bmi", "/bmi/"), ("mental", "/mental-health/"),
                      ("profile", "/accounts/profile/"), ("medication", "/medications/")):
        response = client.get(url, follow=True)
        assert response.status_code == 200, (name, response.status_code)
        (OUT / f"{name}.html").write_text(inline_assets(response.content.decode()), encoding="utf-8")
    transaction.set_rollback(True)

chrome = Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe")
if chrome.exists():
    for name, width, height in (("dashboard", 1440, 1000), ("checkin", 1440, 1000),
                                ("predictions", 1440, 1100), ("predictions", 390, 844)):
        target = OUT / f"{name}-{width}.png"
        command = [str(chrome), "--headless", "--disable-gpu", "--no-first-run",
                   f"--user-data-dir={OUT / 'chrome-profile'}",
                   f"--window-size={width},{height}", "--virtual-time-budget=1500",
                   "--dump-dom", f"--screenshot={target}", (OUT / f"{name}.html").as_uri()]
        completed = subprocess.run(command, timeout=45, capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        size = re.search(r'data-review-width="([^"]+)"', completed.stdout.decode(errors="replace"))
        print(f"Viewport check {name}-{width}: {size[1] if size else 'unavailable'}", flush=True)
        print(f"{target.name}: {'saved' if target.exists() else 'not generated'}", flush=True)
print("Eight synthetic previews rendered. All database changes rolled back.")
