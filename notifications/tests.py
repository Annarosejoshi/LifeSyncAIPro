from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone
from datetime import timedelta

from .models import MedicationRecord


class MedicationRecordViewTests(TestCase):
	def setUp(self):
		self.user = User.objects.create_user(
			username="medication_user",
			password="StrongPass123",
		)
		self.other_user = User.objects.create_user(
			username="other_medication_user",
			password="StrongPass123",
		)
		self.record = MedicationRecord.objects.create(
			user=self.other_user,
			medication_name="Example medication",
			dosage="As prescribed",
			scheduled_date=timezone.localdate(),
			scheduled_time="09:00",
		)

	def test_medication_pages_require_login(self):
		response = self.client.get(
			reverse("notifications:medication_list")
		)

		self.assertEqual(response.status_code, 302)
		self.assertIn("/accounts/login/", response.url)

	def test_create_assigns_record_to_signed_in_user(self):
		self.client.force_login(self.user)
		response = self.client.post(
			reverse("notifications:medication_create"),
			{
				"medication_name": "Daily medication",
				"dosage": "As prescribed",
				"scheduled_date": str(timezone.localdate()),
				"scheduled_time": "08:30",
				"notes": "With breakfast",
			},
		)

		self.assertRedirects(
			response,
			reverse("notifications:medication_list"),
		)
		record = MedicationRecord.objects.get(
			medication_name="Daily medication"
		)
		self.assertEqual(record.user, self.user)
		self.assertFalse(record.taken)

	def test_list_only_shows_signed_in_users_records(self):
		self.client.force_login(self.user)
		MedicationRecord.objects.create(
			user=self.user,
			medication_name="My medication",
			dosage="As prescribed",
			scheduled_date=timezone.localdate(),
			scheduled_time="08:00",
		)

		response = self.client.get(
			reverse("notifications:medication_list")
		)

		self.assertContains(response, "My medication")
		self.assertNotContains(response, "Example medication")

	def test_user_cannot_edit_or_delete_another_users_record(self):
		self.client.force_login(self.user)

		edit_response = self.client.get(
			reverse(
				"notifications:medication_edit",
				args=[self.record.pk],
			)
		)
		delete_response = self.client.post(
			reverse(
				"notifications:medication_delete",
				args=[self.record.pk],
			)
		)

		self.assertEqual(edit_response.status_code, 404)
		self.assertEqual(delete_response.status_code, 404)
		self.assertTrue(
			MedicationRecord.objects.filter(pk=self.record.pk).exists()
		)

	def test_owner_can_edit_and_delete_medication_record(self):
		self.client.force_login(self.other_user)
		edit_response = self.client.post(
			reverse(
				"notifications:medication_edit",
				args=[self.record.pk],
			),
			{
				"medication_name": "Updated medication",
				"dosage": "As prescribed",
				"scheduled_date": str(timezone.localdate()),
				"scheduled_time": "10:30",
				"taken": "on",
				"notes": "Updated reminder",
			},
		)

		self.assertRedirects(
			edit_response,
			reverse("notifications:medication_list"),
		)
		self.record.refresh_from_db()
		self.assertEqual(self.record.medication_name, "Updated medication")
		self.assertTrue(self.record.taken)

		delete_response = self.client.post(
			reverse(
				"notifications:medication_delete",
				args=[self.record.pk],
			)
		)
		self.assertRedirects(
			delete_response,
			reverse("notifications:medication_list"),
		)
		self.assertFalse(
			MedicationRecord.objects.filter(pk=self.record.pk).exists()
		)

	def test_taken_status_requires_post_and_is_user_scoped(self):
		self.client.force_login(self.other_user)
		toggle_url = reverse(
			"notifications:medication_toggle",
			args=[self.record.pk],
		)

		get_response = self.client.get(toggle_url)
		post_response = self.client.post(toggle_url)

		self.record.refresh_from_db()
		self.assertEqual(get_response.status_code, 405)
		self.assertRedirects(
			post_response,
			reverse("notifications:medication_list"),
		)
		self.assertTrue(self.record.taken)

	def test_today_medications_are_available_for_browser_reminders(self):
		self.client.force_login(self.other_user)

		response = self.client.get(
			reverse("notifications:medication_list")
		)

		self.assertEqual(response.status_code, 200)
		self.assertEqual(
			response.context["browser_reminders"][0]["name"],
			self.record.medication_name,
		)

	def test_overdue_unmarked_medication_is_flagged_due(self):
		self.record.scheduled_date = timezone.localdate() - timedelta(days=1)
		self.record.save(update_fields=["scheduled_date"])
		self.client.force_login(self.other_user)

		response = self.client.get(
			reverse("notifications:medication_list")
		)

		self.assertTrue(response.context["records"][0].is_due)
