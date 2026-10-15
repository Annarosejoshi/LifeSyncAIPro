from datetime import datetime, timezone
from types import SimpleNamespace

from django.template.loader import render_to_string
from django.test import SimpleTestCase


class PredictionTimingTests(SimpleTestCase):
    def test_india_time_crosses_midnight_and_distinguishes_saved_time(self):
        html = render_to_string("dashboard/prediction_timing.html", {
            "predictions_calculated_at": datetime(2026, 10, 15, 20, 15, 30, tzinfo=timezone.utc),
            "today_checkin": SimpleNamespace(updated_at=datetime(2026, 10, 15, 19, 0, tzinfo=timezone.utc)),
        })
        self.assertIn("16 Oct 2026, 01:45:30 AM", html)
        self.assertIn("16 Oct 2026, 12:30:00 AM", html)
        self.assertIn("2026-10-16T01:45:30+05:30", html)
        self.assertIn("IST (UTC+05:30)", html)

    def test_unavailable_predictions_do_not_get_a_calculated_time(self):
        html = render_to_string("dashboard/prediction_timing.html", {})
        self.assertNotIn("Results calculated:", html)
        self.assertNotIn("<time", html)
