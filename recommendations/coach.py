import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen


UNSAFE_MODEL_TOPICS = (
    "chest pain", "can't breathe", "cannot breathe", "emergency",
    "suicide", "kill myself", "medicine", "medication", "dose",
    "prescription", "tablet", "diabetes", "diagnose", "diagnosis",
    "disease", "predict", "risk",
)

SUPPORTED_MODEL_TOPICS = (
    "sleep", "tired", "rest", "water", "drink", "hydration",
    "exercise", "activity", "walk", "movement", "fruit", "vegetable",
    "food", "nutrition", "eat", "stress", "anxious", "anxiety", "relax",
    "mood", "sad", "feel", "energy", "score",
)


def answer_wellness_question(question, checkin=None):
    """Return conservative, local wellness guidance for supported topics."""
    text = question.strip().lower()

    if any(term in text for term in ("chest pain", "can't breathe", "cannot breathe", "emergency", "suicide", "kill myself")):
        return (
            "This app cannot handle emergencies. Contact your local emergency "
            "service now, or reach a trusted person nearby."
        )

    if any(term in text for term in ("medicine", "medication", "dose", "prescription", "tablet")):
        return (
            "Follow the directions from your prescribing clinician or pharmacist. "
            "LifeSync cannot recommend starting, stopping, or changing a medicine or dose."
        )

    if any(term in text for term in ("diabetes", "predict", "risk", "disease")):
        return (
            "Disease screening is shown only when its required questionnaire inputs "
            "and compatible trained model are available. A model result is not a "
            "diagnosis; discuss concerns with a qualified healthcare professional."
        )

    if any(term in text for term in ("sleep", "tired", "rest")):
        logged = (
            f"Your latest check-in records {checkin.sleep_hours:g} hours of sleep. "
            if checkin else "You have not recorded a lifestyle check-in yet. "
        )
        return logged + (
            "Many adults aim for at least 7 hours, though needs vary. A regular "
            "sleep and wake time can be a practical place to start."
        )

    if any(term in text for term in ("water", "drink", "hydration")):
        logged = (
            f"Your latest check-in records {checkin.water_liters:g} litres of water. "
            if checkin else "You have not recorded a lifestyle check-in yet. "
        )
        return logged + (
            "Hydration needs vary with the person, activity, climate, and health. "
            "Use your clinician's advice if you have a fluid restriction."
        )

    if any(term in text for term in ("exercise", "activity", "walk", "movement")):
        logged = (
            f"Your latest check-in records {checkin.exercise_minutes} minutes of activity. "
            if checkin else "You have not recorded a lifestyle check-in yet. "
        )
        return logged + (
            "Choose movement that suits your abilities and build gradually. Stop "
            "and seek medical advice if activity causes concerning symptoms."
        )

    if any(term in text for term in ("fruit", "vegetable", "food", "nutrition", "eat")):
        foods = getattr(checkin, "fruit_vegetable_items", []) if checkin else []
        if foods:
            logged = f"You selected {len(foods)} fruit or vegetable item types in your latest check-in. "
        elif checkin:
            logged = "No fruit or vegetable items were selected in your latest check-in. "
        else:
            logged = "You have not recorded a lifestyle check-in yet. "
        return logged + (
            "For your next meal, consider adding a fruit or vegetable you enjoy. "
            "This is general wellness guidance, not an individualized diet plan."
        )

    if any(term in text for term in ("stress", "anxious", "anxiety", "relax")):
        logged = (
            f"Your latest check-in recorded stress level {checkin.stress_level} of 5. "
            if checkin else "You have not recorded a lifestyle check-in yet. "
        )
        return logged + (
            "A short break, slow breathing, or talking with someone you trust may "
            "help. Seek professional support if stress is persistent or affecting daily life."
        )

    if any(term in text for term in ("mood", "sad", "feel", "energy")):
        logged = (
            f"Your latest check-in recorded mood as {checkin.get_mood_display().lower()}. "
            if checkin else "You have not recorded a lifestyle check-in yet. "
        )
        return logged + (
            "A check-in is not a mental-health diagnosis. If low mood persists or "
            "you feel unsafe, contact a qualified professional or someone you trust."
        )

    if any(term in text for term in ("score", "health score")):
        return (
            "The LifeSync score summarizes recorded lifestyle and wellness inputs. "
            "It is an educational project score, not a clinically validated health measure. "
            "Missing sleep-quality information is not treated as a low score."
        )

    return (
        "I can help with general questions about your recorded sleep, water, "
        "exercise, food choices, mood, stress, the lifestyle score, and how "
        "screening works. I cannot diagnose conditions or prescribe treatment. "
        "Try asking about one of those topics."
    )


def answer_with_optional_local_model(question, checkin=None):
    """Use configured local Ollama for safe wellness topics, otherwise rules."""
    fallback = answer_wellness_question(question, checkin)
    model_name = os.environ.get("LIFESYNC_OLLAMA_MODEL", "").strip()
    question_text = question.strip()
    normalized = question_text.lower()

    if not model_name:
        return fallback, "Reviewed local guidance; generative model not configured."

    if (
        any(topic in normalized for topic in UNSAFE_MODEL_TOPICS)
        or not any(topic in normalized for topic in SUPPORTED_MODEL_TOPICS)
    ):
        return fallback, "Safety guidance; not sent to the local model."

    checkin_context = "No recent check-in is available."
    if checkin:
        checkin_context = (
            f"Latest saved check-in: sleep {checkin.sleep_hours} hours; "
            f"water {checkin.water_liters} litres; "
            f"exercise {checkin.exercise_minutes} minutes; "
            f"mood {checkin.get_mood_display()}; "
            f"stress level {checkin.stress_level} of 5."
        )

    prompt = (
        "You are a cautious wellness education assistant. Give one concise, "
        "practical answer using the supplied check-in only when relevant. "
        "Do not diagnose disease, estimate disease risk, prescribe or change "
        "medication, or claim clinical certainty. Mention when needs vary and "
        "suggest qualified care for persistent or concerning symptoms.\n\n"
        f"{checkin_context}\nQuestion: {question_text}"
    )
    request = Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps({
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        answer = payload.get("response", "").strip()
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return fallback, "Reviewed local guidance; local model is unavailable."

    if not answer:
        return fallback, "Reviewed local guidance; local model returned no answer."

    return answer[:3000], f"Local Ollama model: {model_name}"
