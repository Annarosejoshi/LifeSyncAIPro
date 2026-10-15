from django.contrib.auth.models import User
from django.db import models
from django.utils import timezone


class MedicationRecord(models.Model):
	user = models.ForeignKey(
		User,
		on_delete=models.CASCADE,
		related_name="medication_records",
	)
	medication_name = models.CharField(max_length=120)
	dosage = models.CharField(max_length=80)
	scheduled_date = models.DateField(default=timezone.localdate)
	scheduled_time = models.TimeField()
	taken = models.BooleanField(default=False)
	notes = models.CharField(max_length=300, blank=True)
	created_at = models.DateTimeField(auto_now_add=True)
	updated_at = models.DateTimeField(auto_now=True)

	class Meta:
		ordering = ["-scheduled_date", "scheduled_time", "medication_name"]

	def __str__(self):
		return f"{self.medication_name} for {self.user.username}"
