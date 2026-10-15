from django.contrib import admin

# Register your models here.
from django.contrib import admin

from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):

    list_display = (
        "user",
        "prediction_type",
        "risk_level",
        "probability",
        "model_name",
        "created_at",
    )

    list_filter = (
        "prediction_type",
        "risk_level",
        "created_at",
    )

    search_fields = (
        "user__username",
        "user__email",
    )