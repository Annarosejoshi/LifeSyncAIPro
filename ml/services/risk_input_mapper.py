from django.utils import timezone

from accounts.models import UserProfile


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value, default=None):

    if value in [None, ""]:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_age(date_of_birth):

    if not date_of_birth:
        return None

    today = timezone.localdate()

    return (
        today.year
        - date_of_birth.year
        - (
            (today.month, today.day)
            <
            (date_of_birth.month, date_of_birth.day)
        )
    )


def encode_gender(gender):

    if not gender:
        return None

    gender = str(gender).lower()

    # NHANES encoding used in training
    if gender == "male":
        return 1

    if gender == "female":
        return 2

    # Training dataset only had male/female encoding.
    return None


def obesity_flag(bmi):

    bmi = safe_float(bmi)

    if bmi is None:
        return None

    return int(bmi >= 30)


def short_sleep_flag(hours):

    hours = safe_float(hours)

    if hours is None:
        return None

    return int(hours < 7)



# ============================================================
# PROFILE VALUE ENCODERS
# ============================================================

def encode_yes_no_unknown(value):

    if value is None:
        return None

    value = str(value).strip().lower()

    if value == "yes":
        return 1

    if value == "no":
        return 0

    return None


def map_smoking_status(profile):
    """
    Convert the profile smoking choice to the NHANES-style
    encoding used by the trained models.

    0 = Never smoker
    1 = Former smoker
    2 = Current smoker
    """

    if not profile:
        return {
            "ever_smoked_100": None,
            "smoking_status": None,
            "current_smoker": None,
        }

    status = getattr(
        profile,
        "smoking_status",
        None,
    )

    if status:
        status = str(
            status
        ).strip().lower()

    if status == "never":
        return {
            "ever_smoked_100": 0,
            "smoking_status": 0,
            "current_smoker": 0,
        }

    if status == "former":
        return {
            "ever_smoked_100": 1,
            "smoking_status": 1,
            "current_smoker": 0,
        }

    if status == "current":
        return {
            "ever_smoked_100": 1,
            "smoking_status": 2,
            "current_smoker": 1,
        }

    # Backward compatibility with the original Boolean field.
    # True definitely means current smoker.
    # False does NOT tell us whether the person is never/former.
    if getattr(profile, "smoker", None) is True:
        return {
            "ever_smoked_100": 1,
            "smoking_status": 2,
            "current_smoker": 1,
        }

    return {
        "ever_smoked_100": None,
        "smoking_status": None,
        "current_smoker": None,
    }


def map_alcohol_frequency(days_per_year):
    """
    Convert approximate drinking days/year from the profile
    into the categorical ALQ121-style frequency used during
    NHANES model training.

    1  = every day
    2  = nearly every day
    3  = 3-4 times/week
    4  = 2 times/week
    5  = once/week
    6  = 2-3 times/month
    7  = once/month
    8  = 7-11 times/year
    9  = 3-6 times/year
    10 = 1-2 times/year
    """

    days = safe_float(
        days_per_year
    )

    if days is None or days <= 0:
        return None

    if days >= 350:
        return 1

    if days >= 300:
        return 2

    if days >= 156:
        return 3

    if days >= 90:
        return 4

    if days >= 40:
        return 5

    if days >= 20:
        return 6

    if days >= 10:
        return 7

    if days >= 7:
        return 8

    if days >= 3:
        return 9

    return 10


# ============================================================
# GET LATEST RECORDS
# ============================================================

def get_latest_checkin(user):
    """Only today's check-in may supply a daily prediction."""
    try:
        return (
            user.daily_checkins
            .filter(date=timezone.localdate())
            .order_by("-date")
            .first()
        )
    except Exception:
        return None


def get_latest_mental_assessment(user):

    try:
        return (
            user.mental_wellness_assessments
            .order_by("-date")
            .first()
        )
    except Exception:
        return None


# ============================================================
# PROFILE
# ============================================================

def get_profile(user):
    """
    Return the user's health profile directly from the database.

    Using an explicit query avoids relying on Django's cached reverse
    OneToOne descriptor (user.profile) and avoids hiding unrelated
    exceptions behind a broad except block.
    """

    if not user or not getattr(user, "pk", None):
        return None

    return (
        UserProfile.objects
        .filter(user_id=user.pk)
        .first()
    )


# ============================================================
# COMMON FEATURE BUILDER
# ============================================================

def build_common_risk_input(user):

    profile = get_profile(user)

    checkin = get_latest_checkin(user)

    mental = get_latest_mental_assessment(user)


    # --------------------------------------------------------
    # AGE / SEX
    # --------------------------------------------------------

    age = None
    sex = None
    bmi = None

    if profile:

        age = calculate_age(
            profile.date_of_birth
        )

        sex = encode_gender(
            profile.gender
        )

        bmi = profile.bmi


    # --------------------------------------------------------
    # SLEEP
    # --------------------------------------------------------

    sleep_hours = None

    if checkin:

        sleep_hours = safe_float(
            checkin.sleep_hours
        )


    # --------------------------------------------------------
    # ACTIVITY
    # --------------------------------------------------------

    # Use entered values only: one day cannot establish a seven-day total,
    # and screen time is not equivalent to time spent sitting.
    total_activity_minutes_week = safe_float(
        getattr(checkin, "weekly_activity_minutes", None)
    )
    sitting_hours = safe_float(getattr(checkin, "sitting_hours", None))
    sedentary_minutes_day = (
        sitting_hours * 60 if sitting_hours is not None else None
    )


    # --------------------------------------------------------
    # SMOKING
    # --------------------------------------------------------

    smoking = map_smoking_status(
        profile
    )


    # --------------------------------------------------------
    # PROFILE MEASUREMENTS / LAB VALUES
    # --------------------------------------------------------

    waist_circumference = None
    total_cholesterol = None
    hdl_cholesterol = None

    if profile:

        waist_circumference = safe_float(
            getattr(
                profile,
                "waist_circumference",
                None,
            )
        )

        total_cholesterol = safe_float(
            getattr(
                profile,
                "total_cholesterol",
                None,
            )
        )

        hdl_cholesterol = safe_float(
            getattr(
                profile,
                "hdl_cholesterol",
                None,
            )
        )


    # --------------------------------------------------------
    # ALCOHOL
    # --------------------------------------------------------

    alcohol_past_year = None
    alcohol_frequency = None
    average_drinks = None

    if profile:

        alcohol_past_year = (
            encode_yes_no_unknown(
                getattr(
                    profile,
                    "alcohol_past_year",
                    None,
                )
            )
        )

        if alcohol_past_year == 1:

            alcohol_frequency = (
                map_alcohol_frequency(
                    getattr(
                        profile,
                        "alcohol_frequency",
                        None,
                    )
                )
            )

            average_drinks = safe_float(
                getattr(
                    profile,
                    "average_drinks",
                    None,
                )
            )

    # --------------------------------------------------------
    # MENTAL HEALTH
    # --------------------------------------------------------

    mental_symptom_score = None

    # IMPORTANT:
    #
    # Our NHANES model was trained using DPQ/PHQ-type values.
    # MentalWellnessAssessment uses a different 1-5 scale.
    #
    # We should NOT simply add those values and pretend
    # they are equivalent.
    #
    # Therefore leave this as None for V1.
    #
    # Later we can create a properly calibrated mapper/model.


    return {

        # Demographics
        "age":
            age,

        "sex":
            sex,

        "income_ratio":
            None,

        # Body
        "bmi":
            safe_float(bmi),

        "waist_circumference":
            waist_circumference,

        "obesity_flag":
            obesity_flag(bmi),

        # Smoking
        "ever_smoked_100":
            smoking["ever_smoked_100"],

        "smoking_status":
            smoking["smoking_status"],

        "current_smoker":
            smoking["current_smoker"],

        # Activity
        "moderate_sessions_week":
            None,

        "moderate_minutes_session":
            None,

        "vigorous_sessions_week":
            None,

        "vigorous_minutes_session":
            None,

        "total_activity_minutes_week":
            total_activity_minutes_week,

        "sedentary_minutes_day":
            sedentary_minutes_day,

        # Sleep
        "sleep_hours":
            sleep_hours,

        "short_sleep_flag":
            short_sleep_flag(
                sleep_hours
            ),

        # Laboratory values
        "total_cholesterol":
            total_cholesterol,

        "hdl_cholesterol":
            hdl_cholesterol,

        # Alcohol
        "alcohol_past_year":
            alcohol_past_year,

        "alcohol_frequency":
            alcohol_frequency,

        "average_drinks":
            average_drinks,

        # Mental
        "mental_symptom_score":
            mental_symptom_score,
    }


# ============================================================
# DIABETES INPUT
# ============================================================

def build_diabetes_input_from_user(user):

    return build_common_risk_input(
        user
    )


# ============================================================
# HYPERTENSION INPUT
# ============================================================

def build_hypertension_input_from_user(user):

    return build_common_risk_input(
        user
    )


# ============================================================
# CARDIOVASCULAR INPUT
# ============================================================

def build_cardiovascular_input_from_user(user):

    common = build_common_risk_input(
        user
    )

    profile = get_profile(
        user
    )


    # Cardiovascular model has a slightly
    # different feature structure.

    result = {

        "age":
            common.get("age"),

        "sex":
            common.get("sex"),

        "income_ratio":
            common.get("income_ratio"),

        "bmi":
            common.get("bmi"),

        "obesity_flag":
            common.get("obesity_flag"),

        "avg_systolic_bp":
            None,

        "avg_diastolic_bp":
            None,

        "avg_pulse":
            None,

        "high_systolic_flag":
            None,

        "high_diastolic_flag":
            None,

        "total_cholesterol":
            common.get(
                "total_cholesterol"
            ),

        "hdl_cholesterol":
            common.get(
                "hdl_cholesterol"
            ),

        "high_total_cholesterol_flag":
            None,

        "low_hdl_flag":
            None,

        "ever_smoked_100":
            common.get(
                "ever_smoked_100"
            ),

        "smoking_status":
            common.get(
                "smoking_status"
            ),

        "moderate_sessions_week":
            common.get(
                "moderate_sessions_week"
            ),

        "moderate_minutes_session":
            common.get(
                "moderate_minutes_session"
            ),

        "vigorous_sessions_week":
            common.get(
                "vigorous_sessions_week"
            ),

        "vigorous_minutes_session":
            common.get(
                "vigorous_minutes_session"
            ),

        "total_activity_minutes_week":
            common.get(
                "total_activity_minutes_week"
            ),

        "sedentary_minutes_day":
            common.get(
                "sedentary_minutes_day"
            ),
    }


    # ----------------------------------------
    # Profile BP values
    # ----------------------------------------

    if profile:

        systolic = safe_float(
            profile.systolic_bp
        )

        diastolic = safe_float(
            profile.diastolic_bp
        )


        result[
            "avg_systolic_bp"
        ] = systolic

        result[
            "avg_diastolic_bp"
        ] = diastolic


        if systolic is not None:

            result[
                "high_systolic_flag"
            ] = int(
                systolic >= 130
            )


        if diastolic is not None:

            result[
                "high_diastolic_flag"
            ] = int(
                diastolic >= 80
            )


    return result

def build_mental_health_input_from_user(user):

    common = build_common_risk_input(
        user
    )

    return {
        "age":
            common.get("age"),

        "sex":
            common.get("sex"),

        "income_ratio":
            common.get("income_ratio"),

        "bmi":
            common.get("bmi"),

        "waist_circumference":
            common.get("waist_circumference"),

        "obesity_flag":
            common.get("obesity_flag"),

        "ever_smoked_100":
            common.get("ever_smoked_100"),

        "smoking_status":
            common.get("smoking_status"),

        "current_smoker":
            common.get("current_smoker"),

        "moderate_sessions_week":
            common.get("moderate_sessions_week"),

        "moderate_minutes_session":
            common.get("moderate_minutes_session"),

        "vigorous_sessions_week":
            common.get("vigorous_sessions_week"),

        "vigorous_minutes_session":
            common.get("vigorous_minutes_session"),

        "total_activity_minutes_week":
            common.get("total_activity_minutes_week"),

        "sedentary_minutes_day":
            common.get("sedentary_minutes_day"),

        "sleep_hours":
            common.get("sleep_hours"),

        "short_sleep_flag":
            common.get("short_sleep_flag"),

        "total_cholesterol":
            common.get("total_cholesterol"),

        "hdl_cholesterol":
            common.get("hdl_cholesterol"),

        "alcohol_past_year":
            common.get("alcohol_past_year"),

        "alcohol_frequency":
            common.get("alcohol_frequency"),

        "average_drinks":
            common.get("average_drinks"),
    }
