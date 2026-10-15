import hashlib
import json
from pathlib import Path
import re

from django.conf import settings
from django.contrib.auth.models import User
from django.contrib.staticfiles import finders
from django.test import TestCase, SimpleTestCase
from django.urls import reverse


class LocalAssetTests(SimpleTestCase):
    def test_bundled_dependencies_match_recorded_checksums(self):
        root = Path(settings.BASE_DIR) / "static/vendor"
        manifest = json.loads((root / "manifest.json").read_text())
        for name, metadata in manifest.items():
            with self.subTest(asset=name):
                self.assertIsNotNone(finders.find("vendor/" + name))
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), metadata["sha256"])


class OfflinePageAssetTests(TestCase):
    def test_shared_layouts_have_no_remote_style_script_or_font_requests(self):
        self.client.force_login(User.objects.create_user(username="offline_assets"))
        for name in ("dashboard:home", "dashboard:risks", "dashboard:reports", "accounts:profile"):
            html = self.client.get(reverse(name)).content.decode()
            with self.subTest(page=name):
                remote = re.findall(r'<(?:script|link)\b[^>]*(?:src|href)=["\']https?://[^"\']+["\']', html)
                self.assertEqual(remote, [])
                self.assertIn("/static/vendor/bootstrap/bootstrap.min.css", html)
