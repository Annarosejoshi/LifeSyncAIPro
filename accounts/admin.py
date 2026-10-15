from django.contrib import admin

# Register your models here.
from django.contrib import admin

from .models import UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "gender",
        "height_cm",
        "weight_kg",
        "display_bmi",
        "smoker",
        "updated_at",
    )

    search_fields = (
        "user__username",
        "user__email",
        "occupation",
    )

    list_filter = (
        "gender",
        "smoker",
        "diabetes_history",
        "heart_disease_history",
    )

    def display_bmi(self, obj):
        return obj.bmi

    display_bmi.short_description = "BMI"