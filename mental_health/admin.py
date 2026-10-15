from django.contrib import admin

# Register your models here.
from django.contrib import admin

from .models import MentalWellnessAssessment


@admin.register(MentalWellnessAssessment)
class MentalWellnessAssessmentAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "date",
        "mood",
        "stress",
        "anxiety",
        "energy",
        "workload",
    )

    list_filter = (
        "date",
        "stress",
        "anxiety",
    )

    search_fields = (
        "user__username",
        "user__email",
    )

    ordering = (
        "-date",
    )