def map_diabetes_form(cleaned_data):

    return {
        "HighBP":
            float(cleaned_data["high_bp"]),

        "HighChol":
            float(cleaned_data["high_chol"]),

        "CholCheck":
            float(cleaned_data["chol_check"]),

        "BMI":
            float(cleaned_data["bmi"]),

        "Smoker":
            float(cleaned_data["smoker"]),

        "Stroke":
            float(cleaned_data["stroke"]),

        "HeartDiseaseorAttack":
            float(cleaned_data["heart_disease"]),

        "PhysActivity":
            float(cleaned_data["physical_activity"]),

        "Fruits":
            float(cleaned_data["fruits"]),

        "Veggies":
            float(cleaned_data["veggies"]),

        "HvyAlcoholConsump":
            float(cleaned_data["heavy_alcohol"]),

        "AnyHealthcare":
            float(cleaned_data["healthcare"]),

        "NoDocbcCost":
            float(cleaned_data["no_doctor_cost"]),

        "GenHlth":
            float(cleaned_data["general_health"]),

        "MentHlth":
            float(cleaned_data["mental_health_days"]),

        "PhysHlth":
            float(cleaned_data["physical_health_days"]),

        "DiffWalk":
            float(cleaned_data["difficulty_walking"]),

        "Sex":
            float(cleaned_data["sex"]),

        "Age":
            float(cleaned_data["age_category"]),

        "Education":
            float(cleaned_data["education"]),

        "Income":
            float(cleaned_data["income"]),
    }