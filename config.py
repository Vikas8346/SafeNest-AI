"""
SafeNest AI - Phase 1 config
Danger weights for classes already available in the pretrained
YOLO / COCO model (no custom training needed yet).

Scale: 0-100 (used later by the risk engine in Phase 5).
Feel free to add/remove classes as you test.
"""

# Classes we treat as the "child/person" side of the risk equation
PERSON_CLASSES = {"person"}

# Classes we treat as potentially dangerous, with a base danger weight (0-100).
# Includes standard COCO classes (pretrained YOLO) and common hazard aliases.
DANGEROUS_CLASSES = {
    # Sharp & Cutting Tools
    "knife": 95,
    "scissors": 90,
    "sharp tool": 90,
    "tool": 85,
    "fork": 75,

    # Heat, Fire & Gas Hazards
    "open flame": 95,
    "fire": 95,
    "gas stove": 90,
    "stove": 90,
    "burner": 90,
    "gas burner": 90,
    "oven": 85,
    "toaster": 80,
    "microwave": 75,

    # Electrical Hazards
    "electrical socket": 90,
    "socket": 90,
    "outlet": 90,
    "battery": 85,
    "hair drier": 75,

    # Ingestion & Chemical Hazards
    "medicine": 90,
    "pill": 90,
    "pills": 90,
    "chemical": 85,
    "cleaning bottle": 85,
    "chemical bottle": 85,
    "bottle": 75,  # COCO proxy for chemical/cleaning/medicine bottles

    # Minor hazard proxy
    "cell phone": 25,
}

# Everything else we still want to see on screen but don't score as hazard
NEUTRAL_HIGHLIGHT_CLASSES = {
    "teddy bear", "chair", "couch", "bed", "dining table", "cup", "bowl", "book", "clock",
}

# Minimum detection confidence to accept a box at all
CONF_THRESHOLD = 0.35

# Pixel-distance thresholds calibrated for 960x720 camera stream
DIST_HIGH_RISK_PX = 400
DIST_WARNING_PX = 700


def danger_weight(class_name: str) -> int:
    return DANGEROUS_CLASSES.get(class_name.lower(), 0)


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pixel_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def simple_risk_score(danger, distance_px):
    """
    Risk score calculation based on hazard danger weight and proximity.
    Close or medium range to a child/person escalates to full danger level.
    """
    if distance_px <= DIST_HIGH_RISK_PX:
        proximity_factor = 1.0
    elif distance_px <= DIST_WARNING_PX:
        proximity_factor = 0.85
    else:
        proximity_factor = 0.75
    return round(danger * proximity_factor, 1)


def risk_label(score: float) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH RISK"
    if score >= 20:
        return "WARNING"
    return "SAFE"
