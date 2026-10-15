from django.contrib import admin
from .models import MedicationRecord


@admin.register(MedicationRecord)
class MedicationRecordAdmin(admin.ModelAdmin):
	list_display = ("medication_name", "user", "scheduled_date", "scheduled_time", "taken")
	list_filter = ("taken", "scheduled_date")
	search_fields = ("medication_name", "user__username")
