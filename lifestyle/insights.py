def generate_daily_insights(checkin, scores):

    insights = []

    if checkin.sleep_hours < 7:
        insights.append(
            "Your sleep duration is below the recommended target."
        )

    elif checkin.sleep_hours > 9:
        insights.append(
            "Your sleep duration is higher than the typical target range."
        )

    else:
        insights.append(
            "Your sleep duration is within a healthy target range."
        )

    if checkin.water_liters < 2:
        insights.append(
            "Your recorded hydration is relatively low today."
        )

    if checkin.exercise_minutes < 30:
        insights.append(
            "You recorded less than 30 minutes of exercise today."
        )

    else:
        insights.append(
            "You achieved a good amount of physical activity today."
        )

    if checkin.stress_level >= 4:
        insights.append(
            "Your stress level is elevated today."
        )

    if checkin.mood <= 2:
        insights.append(
            "Your mood score is lower than usual wellness targets."
        )

    if checkin.screen_time_hours > 8:
        insights.append(
            "Your screen time is high today."
        )

    if checkin.fruit_vegetable_servings < 3:
        insights.append(
            "Consider increasing fruit and vegetable intake."
        )

    return insights