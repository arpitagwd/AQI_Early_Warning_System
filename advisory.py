"""
advisory.py
Health advisory engine — converts a predicted AQI value into
human-readable warnings, population-specific alerts, and
activity recommendations.
"""
# Each tuple has 4 things: the lower AQI limit, upper limit, the category name, and the hex colour code for that category.
AQI_BANDS = [
    (0,   50,  "Good",        "#00C853"),
    (51,  100, "Satisfactory","#AEEA00"),
    (101, 200, "Moderate",    "#FFD600"),
    (201, 300, "Poor",        "#FF6D00"),
    (301, 400, "Very Poor",   "#DD2C00"),
    (401, 999, "Severe",      "#6A1B9A"),
]

# dictionary, Key = AQI category name, Value = one sentence of advice
GENERAL_ADVICE = {
    "Good":         "Air quality is good. Enjoy outdoor activities freely.",
    "Satisfactory": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
    "Moderate":     "People with respiratory or heart disease, the elderly and children should limit prolonged outdoor exertion.",
    "Poor":         "Everyone may begin to experience health effects. Sensitive groups should avoid outdoor activity.",
    "Very Poor":    "Health alert: everyone may experience serious health effects. Avoid outdoor activity.",
    "Severe":       "Health emergency. Everyone should avoid ALL outdoor activity. Stay indoors with windows closed.",
}

# Nested dictionary — First key is the AQI category, second key is the population group
POPULATION_ADVICE = {
    "Good": {
        "children":  "Safe for outdoor play and school sports.",
        "elderly":   "Safe for normal outdoor activities.",
        "asthma":    "Safe. Keep rescue inhaler handy as a precaution.",
        "healthy":   "No restrictions. Great day for outdoor exercise.",
    },
    "Satisfactory": {
        "children":  "Safe for outdoor play. Limit vigorous activity if they feel discomfort.",
        "elderly":   "Generally safe. Avoid prolonged strenuous outdoor activity.",
        "asthma":    "Monitor symptoms. Carry your inhaler. Limit prolonged exertion.",
        "healthy":   "No significant restrictions.",
    },
    "Moderate": {
        "children":  "Limit outdoor physical activity to short durations. Avoid heavily trafficked areas.",
        "elderly":   "Reduce prolonged or heavy outdoor exertion. Take breaks indoors.",
        "asthma":    "Avoid outdoor exercise. Keep windows closed. Have inhaler accessible.",
        "healthy":   "Unusually sensitive individuals should reduce prolonged outdoor exertion.",
    },
    "Poor": {
        "children":  "Cancel outdoor sports and PE. Keep children indoors during peak hours (10am–4pm).",
        "elderly":   "Stay indoors. Avoid any outdoor physical activity.",
        "asthma":    "Do NOT go outdoors. Use air purifier indoors. Contact doctor if symptoms worsen.",
        "healthy":   "Avoid prolonged outdoor exertion. Consider wearing N95 mask outdoors.",
    },
    "Very Poor": {
        "children":  "Do NOT allow outdoor activity. Schools should cancel all outdoor events.",
        "elderly":   "Stay indoors. Keep all windows shut. Use air purifier.",
        "asthma":    "CRITICAL RISK. Stay indoors. Pre-medicate as per doctor's advice. Seek medical help if needed.",
        "healthy":   "Avoid all outdoor activity. Wear N95 if outdoor exposure unavoidable.",
    },
    "Severe": {
        "children":  "EMERGENCY. Do not go outside under any circumstances.",
        "elderly":   "EMERGENCY. Stay indoors. Seek medical attention proactively.",
        "asthma":    "MEDICAL EMERGENCY RISK. Stay indoors. Call doctor immediately if any symptoms appear.",
        "healthy":   "EMERGENCY. Avoid all outdoor exposure. Wear N95 + goggles if absolutely necessary.",
    },
}


# Heavy breathing activities--become unsafe
# Light activities little okay
ACTIVITY_ADVICE = {
    "Good":         {"jogging": True,  "school_sports": True,  "cycling": True,  "outdoor_dining": True},
    "Satisfactory": {"jogging": True,  "school_sports": True,  "cycling": True,  "outdoor_dining": True},
    "Moderate":     {"jogging": False, "school_sports": False, "cycling": True,  "outdoor_dining": True},
    "Poor":         {"jogging": False, "school_sports": False, "cycling": False, "outdoor_dining": False},
    "Very Poor":    {"jogging": False, "school_sports": False, "cycling": False, "outdoor_dining": False},
    "Severe":       {"jogging": False, "school_sports": False, "cycling": False, "outdoor_dining": False},
}

# These are human-readable labels shown in the app UI.

ACTIVITY_LABELS = {
    "jogging":       "Jogging / running",
    "school_sports": "School outdoor sports",
    "cycling":       "Cycling",
    "outdoor_dining":"Outdoor dining",
}

SPIKE_THRESHOLD = 50   # AQI jump that triggers a spike warning
FESTIVAL_MONTHS = [10, 11]  # Oct–Nov: Diwali + crop-burning season

# based on aqi value. returns a dictionary with the label and colour
def get_bucket(aqi: float) -> dict:
    """Return band info dict for a given AQI value."""
    for low, high, label, color in AQI_BANDS:
        if low <= aqi <= high:
            return {"label": label, "color": color, "low": low, "high": high}
    return {"label": "Severe", "color": "#6A1B9A", "low": 401, "high": 999}

#  called by app.py.
def generate_advisory(aqi: float, prev_aqi: float = None, month: int = None) -> dict:
    """
    Full advisory package for a given AQI reading.

    Parameters
    ----------
    aqi       : predicted AQI value
    prev_aqi  : yesterday's AQI (for spike detection)
    month     : calendar month 1–12 (for festival awareness)

    Returns
    -------
    dict with keys: bucket, general, population, activities,
                    spike_alert, festival_warning, confidence_note
    """
    bucket    = get_bucket(aqi)
    label     = bucket["label"]

    advisory = {
        "bucket":          bucket,
        "general":         GENERAL_ADVICE[label],
        "population":      POPULATION_ADVICE[label],
        "activities":      ACTIVITY_ADVICE[label],
        "activity_labels": ACTIVITY_LABELS,
        "spike_alert":     False,
        "spike_message":   "",
        "festival_warning": False,
        "festival_message": "",
    }

    # Spike detection
    if prev_aqi is not None:
        delta = aqi - prev_aqi
        if delta >= SPIKE_THRESHOLD:
            advisory["spike_alert"]   = True
            # dynamic message(f-- formatted string) 
            advisory["spike_message"] = (
                f"SPIKE ALERT: AQI predicted to jump by {delta:.0f} points "
                f"from {prev_aqi:.0f} to {aqi:.0f}. Unusual pollution event likely."
                #format as float with 0 decimal places
            )

    # Festival / seasonal awareness
    if month in FESTIVAL_MONTHS and aqi > 150:
        advisory["festival_warning"] = True
        advisory["festival_message"] = (
            "Elevated AQI may be linked to seasonal crop burning or festival "
            "firecrackers (Oct–Nov). Pollution may persist for 2–5 days."
        )

    return advisory


def format_activity_row(activity_key: str, allowed: bool) -> str:
    """Return a formatted string for one activity row."""
    icon  = "YES" if allowed else "NO"
    label = ACTIVITY_LABELS[activity_key]
    return f"{icon}  {label}"