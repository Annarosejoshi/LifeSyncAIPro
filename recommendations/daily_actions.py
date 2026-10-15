"""Plain daily habit actions shared by Progress and Wellness Coach."""


def build_daily_actions(checkin):
    if not checkin:
        return []
    missing_meals = [name for name, taken in (
        ("breakfast", checkin.breakfast_taken), ("lunch", checkin.lunch_taken),
        ("dinner", checkin.dinner_taken),
    ) if not taken]
    records = [
        ("🌙", "Sleep", f"{checkin.sleep_hours:g} hours", checkin.sleep_hours < 7,
         "Plan an earlier bedtime tonight.", "Keep a regular bedtime and wake-up time."),
        ("💧", "Water", f"{checkin.water_liters:g} litres", checkin.water_liters < 2,
         "Keep water nearby and drink regularly through the day.", "Continue drinking regularly through the day."),
        ("🚶", "Exercise", f"{checkin.exercise_minutes} minutes", checkin.exercise_minutes < 30,
         "Choose a short walk or another activity that suits your ability.", "Keep movement in your daily routine."),
        ("🥗", "Food variety", f"{checkin.fruit_vegetable_servings} produce choices", checkin.fruit_vegetable_servings < 3,
         "Add a fruit or vegetable to your next meal.", "Keep including different fruits and vegetables."),
        ("🍽️", "Meals", f"{3-len(missing_meals)} of 3 meals recorded", bool(missing_meals),
         "Plan regular meals. If a meal is missing from your log, update your check-in.", "Keep your regular meal routine."),
        ("📱", "Screen breaks", f"{checkin.screen_time_hours:g} hours", checkin.screen_time_hours > 8,
         "Take a screen-free break and reduce late-evening screen use.", "Take regular screen-free breaks."),
        ("💛", "Mental well-being", f"Stress {checkin.stress_level}/5", checkin.stress_level >= 4 or checkin.mood <= 2,
         "Take a quiet break and talk to someone you trust if you need support.", "Make time for an activity you enjoy."),
    ]
    actions = [{"emoji": emoji, "title": title, "value": value, "needs_attention": focus,
                "action": improve if focus else maintain} for emoji, title, value, focus, improve, maintain in records]
    return sorted(actions, key=lambda item: not item["needs_attention"])
