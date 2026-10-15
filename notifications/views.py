from datetime import datetime

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import MedicationRecordForm
from .models import MedicationRecord


@login_required
def medication_list(request):
	today = timezone.localdate()
	now = timezone.localtime()
	records = list(
		MedicationRecord.objects.filter(user=request.user)
	)
	browser_reminders = []

	for record in records:
		scheduled_at = timezone.make_aware(
			datetime.combine(record.scheduled_date, record.scheduled_time),
			timezone.get_current_timezone(),
		)
		record.is_due = not record.taken and scheduled_at <= now

		if record.scheduled_date == today and not record.taken:
			browser_reminders.append({
				"id": record.pk,
				"name": record.medication_name,
				"time": record.scheduled_time.strftime("%I:%M %p").lstrip("0"),
				"scheduledAt": scheduled_at.isoformat(),
			})

	return render(
		request,
		"notifications/medication_list.html",
		{
			"records": records,
			"browser_reminders": browser_reminders,
		},
	)


@login_required
def medication_create(request):
	form = MedicationRecordForm(request.POST or None)
	if request.method == "POST" and form.is_valid():
		record = form.save(commit=False)
		record.user = request.user
		record.save()
		messages.success(request, "Medication record saved.")
		return redirect("notifications:medication_list")

	return render(
		request,
		"notifications/medication_form.html",
		{"form": form, "page_title": "Add medication"},
	)


@login_required
def medication_edit(request, pk):
	record = get_object_or_404(
		MedicationRecord,
		pk=pk,
		user=request.user,
	)
	form = MedicationRecordForm(request.POST or None, instance=record)
	if request.method == "POST" and form.is_valid():
		form.save()
		messages.success(request, "Medication record updated.")
		return redirect("notifications:medication_list")

	return render(
		request,
		"notifications/medication_form.html",
		{"form": form, "page_title": "Edit medication"},
	)


@login_required
@require_POST
def medication_delete(request, pk):
	record = get_object_or_404(
		MedicationRecord,
		pk=pk,
		user=request.user,
	)
	record.delete()
	messages.success(request, "Medication record deleted.")
	return redirect("notifications:medication_list")


@login_required
@require_POST
def medication_toggle(request, pk):
	record = get_object_or_404(
		MedicationRecord,
		pk=pk,
		user=request.user,
	)
	record.taken = not record.taken
	record.save(update_fields=["taken", "updated_at"])
	return redirect("notifications:medication_list")
