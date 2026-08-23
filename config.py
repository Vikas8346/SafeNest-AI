"""
SafeNest AI - Phase 1 config
Danger weights for classes already available in the pretrained
YOLO / COCO model (no custom training needed yet).

Scale: 0-100 (used later by the risk engine in Phase 5).
Feel free to add/remove classes as you test.
"""

# Classes we treat as the "child/person" side of the risk equation
PERSON_CLASSES = {"person"}

# Classes we treat as potentially dangerous, with a base danger weight.
# These all exist in the standard COCO-pretrained YOLO model.
DANGEROUS_CLASSES = {
    "knife": 95,
    "scissors": 70,
    "fork": 30,
    "oven": 60,
    "microwave": 40,
    "toaster": 45,
    "hair drier": 25,   # cord/heat hazard proxy
    "cell phone": 15,   # low-risk placeholder, e.g. choking on small parts
}

# Everything else we still want to see on screen but don't score as risk
NEUTRAL_HIGHLIGHT_CLASSES = {
    "teddy bear", "bottle", "chair", "couch", "bed", "dining table", "cup", "bowl",
}

# Minimum detection confidence to accept a box at all
CONF_THRESHOLD = 0.4

# Pixel-distance thresholds (calibrate per-camera later; these are placeholders)
DIST_HIGH_RISK_PX = 150
DIST_WARNING_PX = 300


def danger_weight(class_name: str) -> int:
    return DANGEROUS_CLASSES.get(class_name, 0)


def box_center(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def pixel_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def simple_risk_score(danger, distance_px):
    """
    Very simple placeholder risk score for Phase 1.
    Phase 5 will replace this with a trained model using
    distance + duration + confidence + danger weight.
    """
    if distance_px <= DIST_HIGH_RISK_PX:
        proximity_factor = 1.0
    elif distance_px <= DIST_WARNING_PX:
        proximity_factor = 0.5
    else:
        proximity_factor = 0.15
    return round(danger * proximity_factor, 1)


def risk_label(score: float) -> str:
    if score >= 70:
        return "CRITICAL"
    if score >= 45:
        return "HIGH RISK"
    if score >= 20:
        return "WARNING"
    return "SAFE"
