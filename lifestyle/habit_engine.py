def evaluate_habit(goal, checkin):

    habit = goal.habit_type
    target = goal.target_value

    if habit == "water":

        actual = checkin.water_liters
        achieved = actual >= target
        unit = "L"

    elif habit == "sleep":

        actual = checkin.sleep_hours
        achieved = actual >= target
        unit = "hours"

    elif habit == "exercise":

        actual = checkin.exercise_minutes
        achieved = actual >= target
        unit = "minutes"

    elif habit == "steps":

        actual = checkin.steps
        achieved = actual >= target
        unit = "steps"

    elif habit == "nutrition":

        actual = checkin.fruit_vegetable_servings
        achieved = actual >= target
        unit = "servings"

    elif habit == "screen_time":

        actual = checkin.screen_time_hours
        achieved = actual <= target
        unit = "hours"

    elif habit == "stress":

        actual = checkin.stress_level
        achieved = actual <= target
        unit = "/5"

    else:

        return None

    return {
        "habit": goal.get_habit_type_display(),
        "target": target,
        "actual": actual,
        "unit": unit,
        "achieved": achieved,
    }