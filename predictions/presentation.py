"""Readable values taken from the exact inputs used for each prediction."""


def overall_screening_summary(risk_cards):
    """Summarize classifications, never average unrelated risk probabilities."""
    available = [
        (title, result) for title, result in risk_cards
        if result and result.get("prediction_available")
        and result.get("prediction") in (0, 1)
    ]
    elevated = [title for title, result in available if result["prediction"] == 1]
    concerns = []
    descriptions = {
        "diabetes": ("Diabetes risk", "The model flagged increased diabetes-related screening risk."),
        "hypertension": ("Blood pressure risk", "The model flagged increased hypertension-related screening risk; it does not measure your blood pressure."),
        "cardiovascular": ("Heart health", "The model flagged increased cardiovascular screening risk."),
        "mental_health": ("Mental wellness", "The model flagged increased depression-related screening risk from your profile and sleep inputs. It does not identify a specific mental health condition or its cause."),
    }
    for label, result in available:
        if result["prediction"] != 1:
            continue
        name, explanation = descriptions.get(result.get("model"), (label, "This screening result is elevated."))
        concerns.append(dict(name=name, explanation=explanation))
    complete = len(available) == 4
    if elevated:
        status = "attention"
        title = f"{concerns[0]['name']} needs attention" if len(concerns) == 1 else "These areas need attention"
        message = "Your screening flagged the following areas for follow-up."
    elif complete:
        status, title = "lower", "No elevated screening results"
        message = "All four results are below their screening cutoffs. Keep tracking your daily habits."
    else:
        status, title = "incomplete", "Overall screening is incomplete"
        message = "Complete the missing details or retry unavailable results to see all four screenings."
    return dict(status=status, title=title, message=message,
                available_count=len(available), elevated_count=len(elevated),
                elevated=elevated, concerns=concerns, complete=complete)


def prediction_input_details(result):
    if not result or not result.get("prediction_available"):
        return []
    details = []
    for feature, value in result.get("input_snapshot", {}).items():
        if value is None:
            continue
        if feature == "age":
            label, text = "Age", f"{value:g} years"
        elif feature == "sex":
            label, text = "Gender", {1: "Male", 2: "Female"}.get(value, "Not recorded")
        elif feature == "bmi":
            label, text = "BMI", f"{value:.2f}"
        elif feature == "smoking_status":
            label, text = "Smoking", {0: "Never", 1: "Former", 2: "Current"}.get(value, "Not recorded")
        elif feature == "sleep_hours":
            label, text = "Sleep today", f"{value:g} hours"
        else:
            continue
        details.append({"label": label, "value": text})
    return details
