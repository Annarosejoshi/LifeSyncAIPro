def generate_mental_wellness_insights(assessment):

    insights = []

    if assessment.stress >= 4:
        insights.append(
            "Your reported stress level is elevated."
        )

    if assessment.anxiety >= 4:
        insights.append(
            "You reported increased anxious feelings today."
        )

    if assessment.energy <= 2:
        insights.append(
            "Your energy level is relatively low."
        )

    if assessment.motivation <= 2:
        insights.append(
            "Your motivation appears lower than your ideal wellness range."
        )

    if assessment.concentration <= 2:
        insights.append(
            "You reported difficulty maintaining concentration."
        )

    if assessment.social_wellbeing <= 2:
        insights.append(
            "Your social wellbeing score is relatively low."
        )

    if assessment.workload >= 4:
        insights.append(
            "Your workload appears high and may be contributing to stress."
        )

    if assessment.mood >= 4 and assessment.stress <= 2:
        insights.append(
            "Your mood and stress indicators look positive today."
        )

    if not insights:
        insights.append(
            "Your reported mental wellness indicators are relatively balanced today."
        )

    return insights