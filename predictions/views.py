from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import render

from ml.services.risk_prediction_service import get_all_health_risks, BRFSS_DIABETES_MODEL_PATH
from ml.services.risk_history_service import save_all_health_risk_history
from ml.services.risk_history_service import normalize_risk_level
from ml.services.risk_prediction_service import predict_diabetes_risk
from .forms import DiabetesRiskForm
from .models import PredictionHistory


@login_required
def diabetes_risk(request):
    risks = get_all_health_risks(request.user)
    form = DiabetesRiskForm(request.POST or None)
    assessment_result = None

    if request.method == "POST" and form.is_valid():
        data = form.cleaned_data
        sex = int(data["sex"])
        smoker = int(data["smoker"])
        age_category = int(data["age_category"])
        age_midpoints = {
            1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52,
            8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 80,
        }

        input_snapshot = {
            "HighBP": int(data["high_bp"]),
            "HighChol": int(data["high_chol"]),
            "CholCheck": int(data["chol_check"]),
            "BMI": float(data["bmi"]),
            "Smoker": smoker,
            "Stroke": int(data["stroke"]),
            "HeartDiseaseorAttack": int(data["heart_disease"]),
            "PhysActivity": int(data["physical_activity"]),
            "Fruits": int(data["fruits"]),
            "Veggies": int(data["veggies"]),
            "HvyAlcoholConsump": int(data["heavy_alcohol"]),
            "AnyHealthcare": int(data["healthcare"]),
            "NoDocbcCost": int(data["no_doctor_cost"]),
            "GenHlth": int(data["general_health"]),
            "MentHlth": int(data["mental_health_days"]),
            "PhysHlth": int(data["physical_health_days"]),
            "DiffWalk": int(data["difficulty_walking"]),
            "Sex": sex,
            "Age": age_category,
            "Education": int(data["education"]),
            "Income": int(data["income"]),
        }

        model_input = {
            **input_snapshot,
            "age": age_midpoints[age_category],
            "sex": 1 if sex == 1 else 2,
            "bmi": float(data["bmi"]),
            "smoking_status": smoker,
        }

        try:
            assessment_result = predict_diabetes_risk(model_input)
        except FileNotFoundError:
            assessment_result = {
                "prediction_available": False,
                "reason": "The optional legacy BRFSS model is not installed. Use Health Predictions for the current diabetes screening.",
            }
        except (RuntimeError, ValueError, OSError):
            assessment_result = {
                "prediction_available": False,
                "reason": "The trained diabetes model could not process this assessment. Verify the model installation and try again.",
            }

        if assessment_result.get("prediction_available"):
            readiness = assessment_result.get("model_input_readiness", {})
            PredictionHistory.objects.create(
                user=request.user,
                prediction_type="diabetes",
                predicted_class=int(assessment_result["prediction"]),
                probability=float(assessment_result["probability"]),
                risk_level=normalize_risk_level(
                    assessment_result.get("risk_level")
                ),
                model_name="BRFSS Random Forest pipeline",
                model_version="2015 three-class",
                threshold=None,
                input_snapshot=input_snapshot,
                explanation=[
                    assessment_result.get("notice", "Experimental screening result."),
                    f"Model features observed: {readiness.get('available_count', 0)}/{readiness.get('total_features', 21)}.",
                ],
            )
            messages.success(request, "Your diabetes screening result was saved to your history.")
        else:
            messages.error(
                request,
                assessment_result.get("reason", "Prediction is unavailable."),
            )

    if request.method != "POST":
        save_all_health_risk_history(request.user, risks)

    form_sections = [
        (
            "Health history",
            [form[name] for name in (
                "high_bp", "high_chol", "chol_check", "stroke",
                "heart_disease", "healthcare", "no_doctor_cost",
            )],
        ),
        (
            "Lifestyle",
            [form[name] for name in (
                "smoker", "physical_activity", "fruits", "veggies",
                "heavy_alcohol",
            )],
        ),
        (
            "Well-being",
            [form[name] for name in (
                "general_health", "mental_health_days", "physical_health_days",
                "difficulty_walking",
            )],
        ),
        (
            "Profile",
            [form[name] for name in (
                "bmi", "sex", "age_category", "education", "income",
            )],
        ),
    ]

    context = {
        "legacy_model_available": BRFSS_DIABETES_MODEL_PATH.is_file(),
        "form": form,
        "form_sections": form_sections,
        "assessment_result": assessment_result,
        "other_risk_results": [
            ("Hypertension", risks.get("hypertension", {})),
            ("Cardiovascular", risks.get("cardiovascular", {})),
            ("Mental wellness", risks.get("mental_health", {})),
        ],
        "diabetes_risk": risks.get("diabetes"),
        "hypertension_risk": risks.get("hypertension"),
        "cardiovascular_risk": risks.get("cardiovascular"),
        "mental_health_risk": risks.get("mental_health"),
    }

    return render(
        request,
        "predictions/diabetes_risk.html",
        context,
    )


@login_required
def prediction_history(request):
    """
    Show saved health-risk predictions
    for the logged-in user.
    """

    history = (
        PredictionHistory.objects
        .filter(user=request.user)
        .order_by("-created_at")
    )

    context = {
        "history": history,
    }

    return render(
        request,
        "predictions/prediction_history.html",
        context,
    )
