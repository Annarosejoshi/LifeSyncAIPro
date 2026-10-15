from django.contrib import admin

# Register your models here.
from django.contrib import admin

from .models import BMIRecord, DailyCheckIn


@admin.register(BMIRecord)
class BMIRecordAdmin(admin.ModelAdmin):
    list_display = ("user", "date", "bmi", "category", "height_cm", "weight_kg")
    list_filter = ("category", "date")
    search_fields = ("user__username", "user__email")
    ordering = ("-date",)


@admin.register(DailyCheckIn)
class DailyCheckInAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "date",
        "sleep_hours",
        "water_liters",
        "exercise_minutes",
        "mood",
        "stress_level",
        "energy_level",
    )

    list_filter = (
        "date",
        "mood",
        "stress_level",
    )

    search_fields = (
        "user__username",
        "user__email",
    )

    ordering = (
        "-date",
    )