def clamp(value, minimum=0, maximum=100):
    return max(minimum, min(value, maximum))


def calculate_mental_wellness_score(assessment):

    mood_score = assessment.mood * 20
    energy_score = assessment.energy * 20
    motivation_score = assessment.motivation * 20
    concentration_score = assessment.concentration * 20
    social_score = assessment.social_wellbeing * 20

    stress_score = (6 - assessment.stress) * 20
    anxiety_score = (6 - assessment.anxiety) * 20
    workload_score = (6 - assessment.workload) * 20

    overall = (
        mood_score * 0.15
        + stress_score * 0.20
        + anxiety_score * 0.15
        + energy_score * 0.10
        + motivation_score * 0.10
        + concentration_score * 0.10
        + social_score * 0.10
        + workload_score * 0.10
    )

    return round(clamp(overall), 1)


def get_mental_wellness_category(score):

    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 55:
        return "Moderate"

    if score >= 40:
        return "Needs Attention"

    return "Low"