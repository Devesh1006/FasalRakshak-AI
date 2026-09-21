from flask import Flask, render_template, request, redirect, url_for
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
import os
import json
import uuid

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter

from translations import (
    LANGUAGES,
    UI,
    t,
    translate_label,
    translate_information
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

UPLOAD_FOLDER = "static/uploads"
MODEL_PATH = "model/crop_disease_model.keras"
CLASS_PATH = "model/classes.json"
REFERENCE_PATH = "model/reference_data.npz"
DATABASE_PATH = "cropcare.db"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# DATABASE
# ============================================================

def init_database():

    conn = sqlite3.connect(DATABASE_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            language TEXT NOT NULL,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            health_score REAL
        )
        """
    )

    # Reports used by the CX0603 hyperlocal outbreak map.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outbreak_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            crop TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            image TEXT NOT NULL,
            evidence_score REAL DEFAULT 0,
            evidence_status TEXT DEFAULT 'unconfirmed',
            independent_cell TEXT DEFAULT '',
            duplicate_of INTEGER,
            photo_confirmed INTEGER DEFAULT 1
        )
        """
    )

    # Safe migration for older databases.
    columns = {
        row[1]
        for row in conn.execute(
            "PRAGMA table_info(outbreak_reports)"
        ).fetchall()
    }

    migrations = {
        "evidence_score":
            "ALTER TABLE outbreak_reports ADD COLUMN evidence_score REAL DEFAULT 0",

        "evidence_status":
            "ALTER TABLE outbreak_reports ADD COLUMN evidence_status TEXT DEFAULT 'unconfirmed'",

        "independent_cell":
            "ALTER TABLE outbreak_reports ADD COLUMN independent_cell TEXT DEFAULT ''",

        "duplicate_of":
            "ALTER TABLE outbreak_reports ADD COLUMN duplicate_of INTEGER",

        "photo_confirmed":
            "ALTER TABLE outbreak_reports ADD COLUMN photo_confirmed INTEGER DEFAULT 1",
    }

    for column, statement in migrations.items():

        if column not in columns:
            conn.execute(statement)

    conn.commit()
    conn.close()


init_database()

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD AI MODEL
# ============================================================

print("Loading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("AI model loaded successfully!")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_PATH,
    "r",
    encoding="utf-8"
) as f:

    class_names = json.load(f)


print(
    "Number of classes:",
    len(class_names)
)


# ============================================================
# LOAD REFERENCE DATA
# ============================================================

if not os.path.exists(REFERENCE_PATH):

    print(
        "ERROR: reference_data.npz not found."
    )

    print(
        "Run: python build_reference.py"
    )

    raise SystemExit


reference_data = np.load(
    REFERENCE_PATH
)

centroids = reference_data["centroids"]

similarity_threshold = float(
    reference_data["similarity_threshold"]
)

print(
    "Reference data loaded."
)

print(
    "Similarity threshold:",
    round(similarity_threshold, 4)
)


# ============================================================
# FIND FEATURE LAYER
# ============================================================

feature_layer = None

for layer in reversed(model.layers):

    if (
        isinstance(layer, tf.keras.layers.Dense)
        and layer.units != len(class_names)
    ):

        feature_layer = layer
        break


if feature_layer is None:

    print(
        "ERROR: Feature layer not found."
    )

    raise SystemExit


feature_model = tf.keras.Model(
    inputs=model.input,
    outputs=feature_layer.output
)


# ============================================================
# RELIABILITY TRANSLATIONS
# ============================================================

RELIABILITY_UI = {

    "en": {
        "consistent":
            "The AI prediction is consistent with the learned image patterns."
    },

    "hi": {
        "consistent":
            "AI भविष्यवाणी सीखे गए छवि पैटर्न के साथ संगत है।"
    },

    "mr": {
        "consistent":
            "AI चा अंदाज शिकलेल्या प्रतिमा नमुन्यांशी सुसंगत आहे."
    }
}


# ============================================================
# CROP HEALTH SCORE TRANSLATIONS
# ============================================================

HEALTH_UI = {

    "en": {

        "title":
            "🌿 Crop Health Score",

        "unavailable":
            "Unable to Calculate",

        "unavailable_text":
            "The AI does not have sufficient evidence for a reliable health assessment.",

        "good":
            "Good",

        "monitor":
            "Monitor",

        "attention":
            "Needs Attention",

        "high_risk":
            "High Risk",

        "reliability":
            "AI Reliability",

        "risk":
            "Disease Risk",

        "low":
            "Low",

        "moderate":
            "Moderate",

        "high":
            "High",

        "quality":
            "Image Quality",

        "good_quality":
            "Good",

        "fair_quality":
            "Needs Improvement",

        "score_note":
            "This score is an AI-based screening indicator, not a confirmed agronomic diagnosis."
    },

    "hi": {

        "title":
            "🌿 फसल स्वास्थ्य स्कोर",

        "unavailable":
            "स्कोर की गणना नहीं की जा सकती",

        "unavailable_text":
            "विश्वसनीय स्वास्थ्य मूल्यांकन के लिए AI के पास पर्याप्त जानकारी नहीं है।",

        "good":
            "अच्छा",

        "monitor":
            "निगरानी रखें",

        "attention":
            "ध्यान देने की आवश्यकता",

        "high_risk":
            "उच्च जोखिम",

        "reliability":
            "AI विश्वसनीयता",

        "risk":
            "रोग जोखिम",

        "low":
            "कम",

        "moderate":
            "मध्यम",

        "high":
            "उच्च",

        "quality":
            "तस्वीर की गुणवत्ता",

        "good_quality":
            "अच्छी",

        "fair_quality":
            "सुधार आवश्यक",

        "score_note":
            "यह स्कोर AI आधारित प्रारंभिक संकेतक है, निश्चित कृषि निदान नहीं।"
    },

    "mr": {

        "title":
            "🌿 पिक आरोग्य स्कोअर",

        "unavailable":
            "स्कोअर मोजता येत नाही",

        "unavailable_text":
            "विश्वसनीय आरोग्य मूल्यांकनासाठी AI कडे पुरेशी माहिती नाही.",

        "good":
            "चांगले",

        "monitor":
            "निगराणी ठेवा",

        "attention":
            "लक्ष देणे आवश्यक",

        "high_risk":
            "उच्च धोका",

        "reliability":
            "AI विश्वासार्हता",

        "risk":
            "रोगाचा धोका",

        "low":
            "कमी",

        "moderate":
            "मध्यम",

        "high":
            "जास्त",

        "quality":
            "फोटोची गुणवत्ता",

        "good_quality":
            "चांगली",

        "fair_quality":
            "सुधारणा आवश्यक",

        "score_note":
            "हा स्कोअर AI आधारित प्राथमिक निर्देशक आहे; हे निश्चित कृषी निदान नाही."
    }
}


# ============================================================
# IMAGE QUALITY
# ============================================================

def check_image_quality(image):

    width, height = image.size

    problems = []

    if width < 250 or height < 250:

        problems.append(
            "The image resolution is low."
        )

    if width * height < 90000:

        problems.append(
            "The image is too small."
        )

    gray = image.convert("L")

    edges = gray.filter(
        ImageFilter.FIND_EDGES
    )

    edge_strength = float(
        np.mean(
            np.array(edges)
        )
    )

    if edge_strength < 12:

        problems.append(
            "The image may be blurry."
        )

    return problems


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    array = np.array(
        image,
        dtype=np.float32
    )

    return np.expand_dims(
        array,
        axis=0
    )


# ============================================================
# AI PREDICTION
# ============================================================

def get_prediction(image):

    return model.predict(
        prepare_image(image),
        verbose=0
    )[0]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def get_feature(image):

    feature = feature_model.predict(
        prepare_image(image),
        verbose=0
    )[0]

    norm = np.linalg.norm(
        feature
    )

    return feature / max(
        norm,
        1e-10
    )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    original = image.convert("RGB")

    view2 = original.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    width, height = original.size

    crop_size = max(
        1,
        int(min(width, height) * 0.85)
    )

    left = (
        width - crop_size
    ) // 2

    top = (
        height - crop_size
    ) // 2

    right = left + crop_size
    bottom = top + crop_size

    view3 = original.crop(
        (
            left,
            top,
            right,
            bottom
        )
    )

    predictions = np.array(
        [
            get_prediction(original),
            get_prediction(view2),
            get_prediction(view3)
        ]
    )

    average_prediction = np.mean(
        predictions,
        axis=0
    )

    sorted_indices = np.argsort(
        average_prediction
    )[::-1]

    top1 = int(
        sorted_indices[0]
    )

    top2 = int(
        sorted_indices[1]
    )

    top3 = int(
        sorted_indices[2]
    )

    confidence = (
        float(
            average_prediction[top1]
        )
        * 100
    )

    top2_confidence = (
        float(
            average_prediction[top2]
        )
        * 100
    )

    top3_confidence = (
        float(
            average_prediction[top3]
        )
        * 100
    )

    margin = (
        confidence
        - top2_confidence
    )

    view_indices = [
        int(np.argmax(p))
        for p in predictions
    ]

    agreement = sum(
        index == top1
        for index in view_indices
    )

    feature = get_feature(
        original
    )

    similarities = np.dot(
        centroids,
        feature
    )

    nearest_reference_index = int(
        np.argmax(similarities)
    )

    return {

        "top1": top1,

        "top2": top2,

        "top3": top3,

        "confidence": confidence,

        "top2_confidence":
            top2_confidence,

        "top3_confidence":
            top3_confidence,

        "margin":
            margin,

        "agreement":
            agreement,

        "predicted_similarity":
            float(
                similarities[top1]
            ),

        "nearest_similarity":
            float(
                similarities[
                    nearest_reference_index
                ]
            ),

        "nearest_reference_index":
            nearest_reference_index
    }


# ============================================================
# RELIABILITY REASONS
# ============================================================

def make_reliability_reason(
    reason,
    lang
):

    mapping = {

        "The image is not sufficiently similar to the training examples for the predicted class.":

        {
            "hi":
                "यह तस्वीर अनुमानित वर्ग के प्रशिक्षण उदाहरणों से पर्याप्त रूप से मेल नहीं खाती।",

            "mr":
                "हा फोटो अंदाज केलेल्या वर्गाच्या प्रशिक्षण उदाहरणांशी पुरेसा जुळत नाही."
        },

        "The classifier prediction and the closest learned class are different.":

        {
            "hi":
                "क्लासिफायर का अनुमान और सबसे निकट सीखा हुआ वर्ग अलग हैं।",

            "mr":
                "क्लासिफायरचा अंदाज आणि सर्वात जवळचा शिकलेला वर्ग वेगळा आहे."
        },

        "The top predictions are very close.":

        {
            "hi":
                "शीर्ष भविष्यवाणियाँ एक-दूसरे के बहुत करीब हैं।",

            "mr":
                "प्रमुख अंदाज एकमेकांच्या खूप जवळ आहेत."
        },

        "The prediction changes when the image is viewed differently.":

        {
            "hi":
                "तस्वीर को अलग तरीके से देखने पर भविष्यवाणी बदल जाती है।",

            "mr":
                "फोटो वेगळ्या पद्धतीने पाहिल्यावर अंदाज बदलतो."
        },

        "The image resolution is low.":

        {
            "hi":
                "तस्वीर का रिज़ॉल्यूशन कम है।",

            "mr":
                "फोटोचे रिझोल्यूशन कमी आहे."
        },

        "The image is too small.":

        {
            "hi":
                "तस्वीर बहुत छोटी है।",

            "mr":
                "फोटो खूप लहान आहे."
        },

        "The image may be blurry.":

        {
            "hi":
                "तस्वीर धुंधली हो सकती है।",

            "mr":
                "फोटो धूसर असू शकतो."
        }
    }

    if lang in mapping.get(
        reason,
        {}
    ):

        return mapping[
            reason
        ][lang]

    return reason


# ============================================================
# CROP HEALTH SCORE
# ============================================================

def calculate_health_score(
    disease,
    confidence,
    predicted_similarity,
    quality_problems
):

    reliability_score = max(
        0.0,
        min(
            100.0,
            predicted_similarity * 100.0
        )
    )

    quality_score = max(
        0.0,
        100.0 - (
            20.0 * len(
                quality_problems
            )
        )
    )

    is_healthy = (
        disease.lower().endswith(
            "___healthy"
        )
    )

    if is_healthy:

        score = (
            0.70 * confidence
            + 0.20 * reliability_score
            + 0.10 * quality_score
        )

    else:

        score = (
            0.70 * (
                100.0 - confidence
            )
            + 0.20 * reliability_score
            + 0.10 * quality_score
        )

    return round(
        max(
            0.0,
            min(
                100.0,
                score
            )
        ),
        1
    )


def get_health_status(
    score,
    language
):

    if language not in HEALTH_UI:
        language = "en"

    if score >= 80:
        key = "good"

    elif score >= 60:
        key = "monitor"

    elif score >= 40:
        key = "attention"

    else:
        key = "high_risk"

    return HEALTH_UI[
        language
    ][key]


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def haversine_km(
    lat1,
    lon1,
    lat2,
    lon2
):

    """Return approximate distance between two GPS points."""

    r = 6371.0

    p1 = np.radians(lat1)
    p2 = np.radians(lat2)

    dp = np.radians(
        lat2 - lat1
    )

    dl = np.radians(
        lon2 - lon1
    )

    a = (
        np.sin(dp / 2) ** 2
        + np.cos(p1)
        * np.cos(p2)
        * np.sin(dl / 2) ** 2
    )

    return float(
        2
        * r
        * np.arcsin(
            np.sqrt(a)
        )
    )


# ============================================================
# OUTBREAK RECOMMENDATIONS
# ============================================================

def get_outbreak_recommendation(
    disease
):

    d = disease.lower()

    if "late_blight" in d:

        return [
            "Inspect nearby plants for rapidly spreading dark lesions.",
            "Remove severely affected plant material and dispose of it safely.",
            "Avoid prolonged leaf wetness and improve field airflow where practical.",
            "Follow locally approved late-blight management guidance before applying any product."
        ]

    if "early_blight" in d:

        return [
            "Inspect lower leaves first and look for expanding brown target-like spots.",
            "Remove badly affected leaves and keep fallen infected debris out of the crop.",
            "Avoid overhead irrigation when practical and maintain good plant spacing.",
            "Use locally approved disease-management practices if symptoms continue to spread."
        ]

    if "apple_scab" in d:

        return [
            "Inspect nearby apple leaves and fruit for olive-brown spots or scabby lesions.",
            "Remove and safely dispose of heavily affected fallen leaves where practical.",
            "Keep foliage as dry as practical and improve canopy airflow.",
            "Follow local extension or agricultural guidance for preventive scab management."
        ]

    if "rust" in d:

        return [
            "Inspect surrounding plants for additional rust-colored pustules or lesions.",
            "Remove severely affected leaves where practical and dispose of them safely.",
            "Improve airflow and avoid unnecessary leaf wetness.",
            "Use locally approved treatment guidance if the outbreak continues."
        ]

    if "powdery_mildew" in d:

        return [
            "Inspect nearby leaves and shoots for expanding white powdery growth.",
            "Remove severely affected plant parts where practical.",
            "Improve airflow and avoid excessive nitrogen or overly dense growth.",
            "Follow locally approved powdery-mildew management guidance."
        ]

    if "bacterial_spot" in d:

        return [
            "Inspect nearby plants for new water-soaked or dark leaf and fruit spots.",
            "Remove severely affected material and avoid moving wet plant debris between plants.",
            "Keep foliage dry where practical and sanitize tools after handling infected material.",
            "Follow locally approved bacterial-disease management guidance."
        ]

    return [
        "Inspect nearby plants for the same symptoms shown in the reported photo.",
        "Separate or safely remove severely affected plant material where practical.",
        "Avoid spreading plant debris, tools, or irrigation water from affected areas.",
        "Consult local agricultural guidance before applying fertilizer or crop-protection products."
    ]


# ============================================================
# OUTBREAK CLUSTERING
# ============================================================

def build_outbreak_clusters(
    reports,
    radius_km=3.0,
    min_reports=3
):

    """Build evidence-weighted hyperlocal disease clusters."""

    clusters = []

    used = set()

    for i, report in enumerate(reports):

        if i in used:
            continue

        cluster = [i]

        used.add(i)

        changed = True

        while changed:

            changed = False

            for j, candidate in enumerate(
                reports
            ):

                if (
                    j in used
                    or candidate["disease"]
                    != report["disease"]
                ):
                    continue

                if any(
                    haversine_km(
                        candidate["latitude"],
                        candidate["longitude"],
                        reports[k]["latitude"],
                        reports[k]["longitude"]
                    ) <= radius_km
                    for k in cluster
                ):

                    cluster.append(j)

                    used.add(j)

                    changed = True

        members = [
            reports[k]
            for k in cluster
        ]

        # Approximately 100m geographic cells.
        cell_groups = {}

        for member in members:

            cell = (
                f"{round(float(member['latitude']), 3):.3f},"
                f"{round(float(member['longitude']), 3):.3f}"
            )

            cell_groups.setdefault(
                cell,
                []
            ).append(member)

        primary_reports = []

        duplicate_count = 0
        supporting_count = 0

        for cell, cell_members in cell_groups.items():

            cell_members.sort(
                key=lambda x:
                float(
                    x.get(
                        "confidence",
                        0
                    )
                ),
                reverse=True
            )

            primary = cell_members[0]

            primary_reports.append(
                primary
            )

            duplicate_count += max(
                0,
                len(cell_members) - 1
            )

            for extra in cell_members[1:]:

                extra[
                    "evidence_status"
                ] = "supporting"

                extra[
                    "independent_cell"
                ] = cell

                extra[
                    "evidence_score"
                ] = round(
                    min(
                        100.0,
                        float(
                            extra.get(
                                "confidence",
                                0
                            )
                        ) * 0.35
                    ),
                    1
                )

                supporting_count += 1

            photo_ok = bool(
                primary.get("image")
            )

            confidence_ok = (
                float(
                    primary.get(
                        "confidence",
                        0
                    )
                ) >= 75.0
            )

            primary[
                "independent_cell"
            ] = cell

            primary[
                "photo_confirmed"
            ] = 1 if photo_ok else 0

            primary[
                "evidence_status"
            ] = (
                "corroborating"
                if (
                    photo_ok
                    and confidence_ok
                )
                else "unconfirmed"
            )

        independent_locations = len(
            cell_groups
        )

        corroborating_count = sum(
            1
            for r in primary_reports
            if r.get(
                "evidence_status"
            ) == "corroborating"
        )

        if primary_reports:

            avg_primary_confidence = (
                sum(
                    float(
                        r.get(
                            "confidence",
                            0
                        )
                    )
                    for r in primary_reports
                )
                / len(primary_reports)
            )

        else:

            avg_primary_confidence = 0.0

        independence_component = (
            min(
                independent_locations / 5.0,
                1.0
            )
            * 100.0
        )

        evidence_score = (
            0.55
            * avg_primary_confidence
            + 0.45
            * independence_component
        )

        evidence_score = round(
            max(
                0.0,
                min(
                    100.0,
                    evidence_score
                )
            ),
            1
        )

        # Alert severity.
        if (
            corroborating_count >= 6
            and independent_locations >= 4
            and evidence_score >= 80
        ):

            severity = "High Risk"
            severity_level = "high"

        elif (
            corroborating_count >= 4
            and independent_locations >= 3
            and evidence_score >= 70
        ):

            severity = "Moderate Risk"
            severity_level = "moderate"

        elif (
            corroborating_count >= 3
            and independent_locations >= 2
            and evidence_score >= 60
        ):

            severity = "Watch Area"
            severity_level = "watch"

        else:

            continue

        avg_lat = (
            sum(
                float(
                    m["latitude"]
                )
                for m in primary_reports
            )
            / len(primary_reports)
        )

        avg_lon = (
            sum(
                float(
                    m["longitude"]
                )
                for m in primary_reports
            )
            / len(primary_reports)
        )

        avg_conf = (
            sum(
                float(
                    m["confidence"]
                )
                for m in primary_reports
            )
            / len(primary_reports)
        )

        for member in members:

            if member in primary_reports:

                member[
                    "evidence_status"
                ] = (
                    "corroborating"
                    if (
                        float(
                            member.get(
                                "confidence",
                                0
                            )
                        ) >= 75
                        and bool(
                            member.get(
                                "image"
                            )
                        )
                    )
                    else "unconfirmed"
                )

                member[
                    "evidence_score"
                ] = round(
                    min(
                        100.0,
                        float(
                            member.get(
                                "confidence",
                                0
                            )
                        )
                        * 0.55
                        + 45.0
                    ),
                    1
                )

        clusters.append(
            {
                "disease":
                    members[0]["disease"],

                "crop":
                    members[0]["crop"],

                "count":
                    len(members),

                "submitted_count":
                    len(members),

                "locations":
                    independent_locations,

                "independent_locations":
                    independent_locations,

                "corroborating_count":
                    corroborating_count,

                "supporting_count":
                    supporting_count,

                "duplicate_count":
                    duplicate_count,

                "latitude":
                    round(avg_lat, 6),

                "longitude":
                    round(avg_lon, 6),

                "confidence":
                    round(avg_conf, 1),

                "evidence_score":
                    evidence_score,

                "severity":
                    severity,

                "severity_level":
                    severity_level,

                "alert":
                    True
            }
        )

    return clusters


# ============================================================
# REPORT OUTBREAK
# ============================================================

@app.route(
    "/report",
    methods=["POST"]
)
def report_outbreak():

    data = (
        request.get_json(
            silent=True
        )
        or request.form
    )

    try:

        latitude = float(
            data.get("latitude")
        )

        longitude = float(
            data.get("longitude")
        )

        confidence = float(
            data.get(
                "confidence",
                0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        return {
            "ok": False,
            "message":
                "Invalid location or confidence."
        }, 400

    if not (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):

        return {
            "ok": False,
            "message":
                "Invalid GPS coordinates."
        }, 400

    crop = str(
        data.get(
            "crop",
            "Unknown"
        )
    )[:100]

    disease = str(
        data.get(
            "disease",
            "Unknown"
        )
    )[:150]

    image = str(
        data.get(
            "image",
            ""
        )
    )[:255]

    if not image:

        return {
            "ok": False,
            "message":
                "Photo reference missing."
        }, 400

    photo_confirmed = (
        1
        if confidence >= 75
        else 0
    )

    cell = (
        f"{round(latitude, 3):.3f},"
        f"{round(longitude, 3):.3f}"
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    recent = conn.execute(
        """
        SELECT *
        FROM outbreak_reports
        WHERE disease = ?
        AND created_at >= datetime(
            'now',
            '-30 minutes'
        )
        ORDER BY id DESC
        """,
        (disease,)
    ).fetchall()

    for row in recent:

        if (
            haversine_km(
                latitude,
                longitude,
                row["latitude"],
                row["longitude"]
            ) < 0.1
        ):

            conn.close()

            return {
                "ok": True,
                "duplicate": True,
                "message":
                    "A recent nearby report already exists; it was not counted as a new location."
            }

    base_evidence = round(
        min(
            100.0,
            max(
                0.0,
                confidence * 0.70
            )
        ),
        1
    )

    initial_status = (
        "corroborating"
        if photo_confirmed
        else "unconfirmed"
    )

    conn.execute(
        """
        INSERT INTO outbreak_reports
        (
            created_at,
            crop,
            disease,
            confidence,
            latitude,
            longitude,
            image,
            evidence_score,
            evidence_status,
            independent_cell,
            duplicate_of,
            photo_confirmed
        )
        VALUES (
            datetime('now'),
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, NULL, ?
        )
        """,
        (
            crop,
            disease,
            confidence,
            latitude,
            longitude,
            image,
            base_evidence,
            initial_status,
            cell,
            photo_confirmed
        )
    )

    conn.commit()
    conn.close()

    return {
        "ok": True,
        "duplicate": False,
        "message":
            "Photo report added. Geographic independence will determine how strongly it contributes to an outbreak."
    }


# ============================================================
# DEMO OUTBREAK
# ============================================================

@app.route(
    "/demo-outbreak",
    methods=["GET", "POST"]
)
def demo_outbreak():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    demo_points = [

        (96.5, 19.0520, 73.0147),
        (94.2, 19.0640, 73.0240),
        (95.1, 19.0415, 73.0060),
        (93.8, 19.0570, 73.0350),
        (96.1, 19.0320, 73.0200),
        (92.9, 19.0700, 73.0010)
    ]

    for (
        confidence,
        latitude,
        longitude
    ) in demo_points:

        conn.execute(
            """
            INSERT INTO outbreak_reports
            (
                created_at,
                crop,
                disease,
                confidence,
                latitude,
                longitude,
                image,
                evidence_score,
                evidence_status,
                independent_cell,
                photo_confirmed
            )
            VALUES (
                datetime('now'),
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                "Apple (Demo)",
                "Apple___Apple_scab",
                confidence,
                latitude,
                longitude,
                "demo-photo-confirmed",
                round(
                    confidence * 0.70,
                    1
                ),
                "corroborating",
                (
                    f"{round(latitude, 3):.3f},"
                    f"{round(longitude, 3):.3f}"
                ),
                1
            )
        )

    conn.commit()
    conn.close()

    return {
        "ok": True,
        "message":
            "6 synthetic photo-confirmed reports from independent locations added."
    }


# ============================================================
# DEMO PANIC REPORTING
# ============================================================

@app.route(
    "/demo-panic",
    methods=["GET", "POST"]
)
def demo_panic():

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    base_lat = 19.0520
    base_lon = 73.0147

    for i in range(10):

        latitude = (
            base_lat
            + (i % 3) * 0.00015
        )

        longitude = (
            base_lon
            + (i % 4) * 0.00015
        )

        confidence = (
            92.0
            + (i % 5)
        )

        conn.execute(
            """
            INSERT INTO outbreak_reports
            (
                created_at,
                crop,
                disease,
                confidence,
                latitude,
                longitude,
                image,
                evidence_score,
                evidence_status,
                independent_cell,
                photo_confirmed
            )
            VALUES (
                datetime('now'),
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?
            )
            """,
            (
                "Apple (Panic Demo)",
                "Apple___Apple_scab",
                confidence,
                latitude,
                longitude,
                "panic-copycat-photo",
                round(
                    confidence * 0.35,
                    1
                ),
                "supporting",
                (
                    f"{round(latitude, 3):.3f},"
                    f"{round(longitude, 3):.3f}"
                ),
                1
            )
        )

    conn.commit()
    conn.close()

    return {
        "ok": True,
        "message":
            "10 copycat panic reports added from one small area; they should not create strong independent evidence."
    }


# ============================================================
# NEARBY ALERT
# ============================================================

@app.route(
    "/nearby-alert",
    methods=["POST"]
)
def nearby_alert():

    data = (
        request.get_json(
            silent=True
        )
        or request.form
    )

    try:

        latitude = float(
            data.get("latitude")
        )

        longitude = float(
            data.get("longitude")
        )

        radius_km = float(
            data.get(
                "radius_km",
                5
            )
        )

    except (
        TypeError,
        ValueError
    ):

        return {
            "ok": False,
            "message":
                "Invalid location."
        }, 400

    if not (
        -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):

        return {
            "ok": False,
            "message":
                "Invalid GPS coordinates."
        }, 400

    radius_km = max(
        1.0,
        min(
            radius_km,
            10.0
        )
    )

    disease = str(
        data.get(
            "disease",
            ""
        )
    ).strip()[:150]

    if not disease:

        return {
            "ok": False,
            "message":
                "Disease information is missing."
        }, 400

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT
            id,
            created_at,
            crop,
            disease,
            confidence,
            latitude,
            longitude,
            image,
            evidence_score,
            evidence_status,
            independent_cell,
            duplicate_of,
            photo_confirmed
        FROM outbreak_reports
        WHERE created_at >= datetime(
            'now',
            '-30 days'
        )
        AND disease = ?
        ORDER BY id DESC
        """,
        (disease,)
    ).fetchall()

    conn.close()

    reports = [
        dict(row)
        for row in rows
    ]

    clusters = build_outbreak_clusters(
        reports
    )

    nearby = []

    for cluster in clusters:

        distance = haversine_km(
            latitude,
            longitude,
            float(
                cluster["latitude"]
            ),
            float(
                cluster["longitude"]
            )
        )

        if distance <= radius_km:

            item = dict(
                cluster
            )

            item[
                "distance_km"
            ] = round(
                distance,
                2
            )

            nearby.append(
                item
            )

    nearby.sort(
        key=lambda item: (
            item["distance_km"],
            -float(
                item[
                    "evidence_score"
                ]
            )
        )
    )

    nearby_report_count = sum(
        1
        for report in reports
        if haversine_km(
            latitude,
            longitude,
            float(
                report["latitude"]
            ),
            float(
                report["longitude"]
            )
        ) <= radius_km
    )

    if nearby:

        return {
            "ok": True,
            "alert": True,
            "radius_km":
                radius_km,
            "alerts":
                nearby[:3],
            "nearby_report_count":
                nearby_report_count
        }

    return {
        "ok": True,
        "alert": False,
        "radius_km":
            radius_km,
        "alerts": [],
        "nearby_report_count":
            nearby_report_count
    }


# ============================================================
# DEMO NEARBY ALERT
# ============================================================

@app.route(
    "/demo-nearby-alert",
    methods=["GET", "POST"]
)
def demo_nearby_alert():

    demo_reports = [

        {
            "id": 1001,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                96.5,
            "latitude":
                19.0520,
            "longitude":
                73.0147,
            "image":
                "demo-photo-1",
            "evidence_score":
                67.6,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.052,73.015",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        },

        {
            "id": 1002,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                94.2,
            "latitude":
                19.0640,
            "longitude":
                73.0240,
            "image":
                "demo-photo-2",
            "evidence_score":
                65.9,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.064,73.024",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        },

        {
            "id": 1003,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                95.1,
            "latitude":
                19.0415,
            "longitude":
                73.0060,
            "image":
                "demo-photo-3",
            "evidence_score":
                66.6,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.042,73.006",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        },

        {
            "id": 1004,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                93.8,
            "latitude":
                19.0570,
            "longitude":
                73.0350,
            "image":
                "demo-photo-4",
            "evidence_score":
                65.7,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.057,73.035",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        },

        {
            "id": 1005,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                96.1,
            "latitude":
                19.0320,
            "longitude":
                73.0200,
            "image":
                "demo-photo-5",
            "evidence_score":
                67.3,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.032,73.020",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        },

        {
            "id": 1006,
            "created_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "crop":
                "Apple (Demo)",
            "disease":
                "Apple___Apple_scab",
            "confidence":
                92.9,
            "latitude":
                19.0700,
            "longitude":
                73.0010,
            "image":
                "demo-photo-6",
            "evidence_score":
                65.0,
            "evidence_status":
                "corroborating",
            "independent_cell":
                "19.070,73.001",
            "duplicate_of":
                None,
            "photo_confirmed":
                1
        }
    ]

    independent_locations = len(
        {
            report[
                "independent_cell"
            ]
            for report in demo_reports
        }
    )

    photo_confirmed = sum(
        1
        for report in demo_reports
        if report[
            "photo_confirmed"
        ] == 1
    )

    duplicates = sum(
        1
        for report in demo_reports
        if report[
            "evidence_status"
        ] == "supporting"
    )

    average_confidence = (
        sum(
            report[
                "confidence"
            ]
            for report in demo_reports
        )
        / len(demo_reports)
    )

    independence_component = (
        min(
            independent_locations / 5.0,
            1.0
        )
        * 100.0
    )

    evidence_score = (
        0.55
        * average_confidence
        + 0.45
        * independence_component
    )

    evidence_score = round(
        min(
            100.0,
            max(
                0.0,
                evidence_score
            )
        ),
        1
    )

    severity = "High Risk"

    return {

        "ok":
            True,

        "demo":
            True,

        "alert":
            True,

        "disease":
            "Apple___Apple_scab",

        "crop":
            "Apple",

        "reports":
            demo_reports,

        "independent_locations":
            independent_locations,

        "photo_confirmed":
            photo_confirmed,

        "duplicates":
            duplicates,

        "average_confidence":
            round(
                average_confidence,
                1
            ),

        "evidence_score":
            evidence_score,

        "severity":
            severity,

        "message":
            "Demo alert generated from six independent photo-confirmed locations."
    }


# ============================================================
# OUTBREAK MAP
# ============================================================

@app.route("/outbreak-map")
def outbreak_map():

    language = request.args.get(
        "lang",
        "en"
    )

    if language not in LANGUAGES:
        language = "en"

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT
            id,
            created_at,
            crop,
            disease,
            confidence,
            latitude,
            longitude,
            image,
            evidence_score,
            evidence_status,
            independent_cell,
            duplicate_of,
            photo_confirmed
        FROM outbreak_reports
        WHERE created_at >= datetime(
            'now',
            '-30 days'
        )
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    reports = [
        dict(row)
        for row in rows
    ]

    clusters = build_outbreak_clusters(
        reports
    )

    disease_counts = {}
    crop_counts = {}

    for report in reports:

        disease_counts[
            report["disease"]
        ] = disease_counts.get(
            report["disease"],
            0
        ) + 1

        crop_counts[
            report["crop"]
        ] = crop_counts.get(
            report["crop"],
            0
        ) + 1

    most_reported_disease = (
        max(
            disease_counts,
            key=disease_counts.get
        )
        if disease_counts
        else "--"
    )

    highest_risk_crop = (
        max(
            crop_counts,
            key=crop_counts.get
        )
        if crop_counts
        else "--"
    )

    trend_counts = {}

    for report in reports:

        day = str(
            report[
                "created_at"
            ]
        )[:10]

        trend_counts[day] = (
            trend_counts.get(
                day,
                0
            )
            + 1
        )

    trend = [

        {
            "date": day,
            "count":
                trend_counts[day]
        }

        for day in sorted(
            trend_counts
        )
    ]

    for cluster in clusters:

        cluster[
            "recommendations"
        ] = get_outbreak_recommendation(
            cluster["disease"]
        )

    map_ui = {

        "en": {

            "title":
                "🗺️ Hyperlocal Disease Alert Map",

            "subtitle":
                "Evidence-weighted photo reports clustered by location",

            "reports":
                "Recent Photo Reports",

            "alerts":
                "Active Hyperlocal Alerts",

            "no_alerts":
                "No evidence-weighted cluster alert yet. Independent photo-confirmed locations are required.",

            "back":
                "← Back to CropCare AI",

            "history":
                "📊 Monitoring History",

            "legend":
                "Alert logic: independent photo-confirmed locations matter more than repeated reports from the same area.",

            "privacy":
                "Location is used only for this demo's outbreak map.",

            "stats_reports":
                "Photo Reports",

            "stats_alerts":
                "Active Outbreaks",

            "stats_disease":
                "Most Reported Disease",

            "stats_crop":
                "Most Reported Crop",

            "risk":
                "Risk",

            "locations":
                "independent locations",

            "confidence":
                "AI confidence",

            "empty":
                "No photo reports yet. Run a crop scan and add a local report.",

            "high":
                "High Risk",

            "moderate":
                "Moderate Risk",

            "watch":
                "Watch Area",

            "recommendations":
                "Recommended Farmer Actions",

            "trend_title":
                "📈 Outbreak Report Trend",

            "trend_empty":
                "Not enough date history for a trend yet.",

            "evidence":
                "Evidence Score",

            "corroborating":
                "Corroborating",

            "supporting":
                "Supporting",

            "duplicates":
                "Same-area reports",

            "logic":
                "Independent photo-confirmed evidence is weighted more strongly than raw report count.",

            "panic":
                "Simulate Panic Reporting",

            "genuine":
                "Simulate Genuine Outbreak"
        },

        "hi": {

            "title":
                "🗺️ स्थानीय रोग चेतावनी मानचित्र",

            "subtitle":
                "स्थान के अनुसार प्रमाण-आधारित फोटो रिपोर्ट का समूह",

            "reports":
                "हाल की फोटो रिपोर्ट",

            "alerts":
                "सक्रिय स्थानीय चेतावनियाँ",

            "no_alerts":
                "अभी कोई प्रमाण-आधारित क्लस्टर चेतावनी नहीं है। अलग-अलग स्थानों से फोटो पुष्टि आवश्यक है।",

            "back":
                "← CropCare AI पर वापस",

            "history":
                "📊 निगरानी इतिहास",

            "legend":
                "अलग-अलग स्थानों की फोटो-पुष्ट रिपोर्ट को एक ही क्षेत्र की दोहराई गई रिपोर्ट से अधिक महत्व मिलता है।",

            "privacy":
                "स्थान का उपयोग केवल इस डेमो के रोग मानचित्र के लिए किया जाता है।",

            "stats_reports":
                "फोटो रिपोर्ट",

            "stats_alerts":
                "सक्रिय प्रकोप",

            "stats_disease":
                "सबसे अधिक रिपोर्ट किया गया रोग",

            "stats_crop":
                "सबसे अधिक रिपोर्ट की गई फसल",

            "risk":
                "जोखिम",

            "locations":
                "स्वतंत्र स्थान",

            "confidence":
                "AI विश्वास",

            "empty":
                "अभी कोई फोटो रिपोर्ट नहीं है। फसल स्कैन करके स्थानीय रिपोर्ट जोड़ें।",

            "high":
                "उच्च जोखिम",

            "moderate":
                "मध्यम जोखिम",

            "watch":
                "निगरानी क्षेत्र",

            "recommendations":
                "किसानों के लिए सुझाए गए कदम",

            "trend_title":
                "📈 प्रकोप रिपोर्ट रुझान",

            "trend_empty":
                "रुझान के लिए अभी पर्याप्त दिन का इतिहास नहीं है।",

            "evidence":
                "प्रमाण स्कोर",

            "corroborating":
                "पुष्टि करने वाली",

            "supporting":
                "सहायक",

            "duplicates":
                "एक ही क्षेत्र की रिपोर्ट",

            "logic":
                "अलग-अलग स्थानों की फोटो-पुष्ट रिपोर्ट को कच्ची रिपोर्ट संख्या से अधिक महत्व दिया जाता है।",

            "panic":
                "घबराहट वाली रिपोर्ट का डेमो",

            "genuine":
                "वास्तविक प्रकोप का डेमो"
        },

        "mr": {

            "title":
                "🗺️ स्थानिक रोग इशारा नकाशा",

            "subtitle":
                "स्थानानुसार पुराव्यावर आधारित फोटो अहवालांचे गट",

            "reports":
                "अलीकडील फोटो अहवाल",

            "alerts":
                "सक्रिय स्थानिक इशारे",

            "no_alerts":
                "अजून पुराव्यावर आधारित क्लस्टर इशारा नाही. वेगवेगळ्या ठिकाणांहून फोटो पुष्टी आवश्यक आहे.",

            "back":
                "← CropCare AI कडे परत",

            "history":
                "📊 निरीक्षण इतिहास",

            "legend":
                "वेगवेगळ्या ठिकाणांहून आलेल्या फोटो-पुष्ट अहवालांना एकाच भागातील पुनरावृत्तीच्या अहवालांपेक्षा जास्त महत्त्व दिले जाते.",

            "privacy":
                "स्थानाचा वापर फक्त या डेमोच्या रोग नकाशासाठी केला जातो.",

            "stats_reports":
                "फोटो अहवाल",

            "stats_alerts":
                "सक्रिय प्रादुर्भाव",

            "stats_disease":
                "सर्वाधिक नोंदलेला रोग",

            "stats_crop":
                "सर्वाधिक नोंदलेले पीक",

            "risk":
                "धोका",

            "locations":
                "स्वतंत्र ठिकाणे",

            "confidence":
                "AI विश्वास",

            "empty":
                "अजून फोटो अहवाल नाहीत. पीक स्कॅन करून स्थानिक अहवाल जोडा.",

            "high":
                "उच्च धोका",

            "moderate":
                "मध्यम धोका",

            "watch":
                "निगराणी क्षेत्र",

            "recommendations":
                "शेतकऱ्यांसाठी सुचवलेली कृती",

            "trend_title":
                "📈 प्रादुर्भाव अहवाल कल",

            "trend_empty":
                "कल दाखवण्यासाठी अजून पुरेसा दिवसांचा इतिहास नाही.",

            "evidence":
                "पुरावा स्कोअर",

            "corroborating":
                "पुष्टी करणारे",

            "supporting":
                "सहाय्यक",

            "duplicates":
                "त्याच भागातील अहवाल",

            "logic":
                "वेगवेगळ्या ठिकाणांहून आलेल्या फोटो-पुष्ट पुराव्यांना कच्च्या अहवाल संख्येपेक्षा जास्त वजन दिले जाते.",

            "panic":
                "घबराट अहवाल डेमो",

            "genuine":
                "प्रत्यक्ष प्रादुर्भाव डेमो"
        }
    }[language]

    return render_template(
        "outbreak_map.html",

        language=language,

        languages=LANGUAGES,

        map_ui=map_ui,

        reports=reports,

        clusters=clusters,

        total_reports=len(
            reports
        ),

        active_alerts=len(
            clusters
        ),

        most_reported_disease=
            most_reported_disease,

        highest_risk_crop=
            highest_risk_crop,

        trend=trend
    )


# ============================================================
# HISTORY
# ============================================================

@app.route("/history")
def history():

    language = request.args.get(
        "lang",
        "en"
    )

    if language not in LANGUAGES:
        language = "en"

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT * FROM history ORDER BY id DESC"
    ).fetchall()

    conn.close()

    health_ui = HEALTH_UI[
        language
    ]

    history_ui = {

        "title": {
            "en":
                "📊 Crop Monitoring History",

            "hi":
                "📊 फसल निगरानी इतिहास",

            "mr":
                "📊 पिक निरीक्षण इतिहास"
        }[language],

        "total": {
            "en":
                "Total Scans",

            "hi":
                "कुल स्कैन",

            "mr":
                "एकूण स्कॅन"
        }[language],

        "average": {
            "en":
                "Average Health Score",

            "hi":
                "औसत स्वास्थ्य स्कोर",

            "mr":
                "सरासरी आरोग्य स्कोअर"
        }[language],

        "healthy": {
            "en":
                "Healthy Results",

            "hi":
                "स्वस्थ परिणाम",

            "mr":
                "निरोगी परिणाम"
        }[language],

        "disease": {
            "en":
                "Disease Detected",

            "hi":
                "रोग का पता चला",

            "mr":
                "रोग आढळला"
        }[language],

        "date": {
            "en":
                "Date",

            "hi":
                "दिनांक",

            "mr":
                "दिनांक"
        }[language],

        "crop":
            UI[language]["crop"],

        "disease_label":
            UI[language]["disease"],

        "confidence": {
            "en":
                "Confidence",

            "hi":
                "विश्वास",

            "mr":
                "विश्वासार्हता"
        }[language],

        "score": {
            "en":
                "Health Score",

            "hi":
                "स्वास्थ्य स्कोर",

            "mr":
                "आरोग्य स्कोअर"
        }[language],

        "empty": {
            "en":
                "No monitoring records yet.",

            "hi":
                "अभी कोई निगरानी रिकॉर्ड नहीं है।",

            "mr":
                "अजून कोणतेही निरीक्षण रेकॉर्ड नाहीत."
        }[language],

        "back": {
            "en":
                "← Check Another Crop",

            "hi":
                "← दूसरी फसल जांचें",

            "mr":
                "← दुसरे पीक तपासा"
        }[language],

        "clear": {
            "en":
                "Clear History",

            "hi":
                "इतिहास साफ करें",

            "mr":
                "इतिहास साफ करा"
        }[language],

        "uncertain":
            HEALTH_UI[
                language
            ]["unavailable"]
    }

    total = len(
        rows
    )

    scores = [
        r["health_score"]
        for r in rows
        if r["health_score"] is not None
    ]

    average = (
        round(
            sum(scores)
            / len(scores),
            1
        )
        if scores
        else "--"
    )

    healthy_count = sum(
        1
        for r in rows
        if "healthy"
        in r["disease"].lower()
    )

    disease_count = (
        total
        - healthy_count
    )

    chart_data = []

    for r in reversed(
        rows
    ):

        score = r[
            "health_score"
        ]

        if score is not None:

            chart_data.append(
                {
                    "date":
                        str(
                            r["created_at"]
                        ),

                    "score":
                        float(score)
                }
            )

    return render_template(
        "history.html",

        language=language,

        languages=LANGUAGES,

        ui=UI[language],

        health_ui=health_ui,

        history_ui=history_ui,

        rows=rows,

        total=total,

        average=average,

        healthy_count=
            healthy_count,

        disease_count=
            disease_count,

        chart_data=
            chart_data
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/clear-history",
    methods=["POST"]
)
def clear_history():

    language = request.form.get(
        "language",
        "en"
    )

    if language not in LANGUAGES:
        language = "en"

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.execute(
        "DELETE FROM history"
    )

    conn.commit()
    conn.close()

    return redirect(
        url_for(
            "history",
            lang=language
        )
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    language = request.args.get(
        "lang",
        "en"
    )

    if language not in LANGUAGES:
        language = "en"

    return render_template(
        "index.html",

        language=language,

        languages=LANGUAGES,

        ui=UI[language]
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    language = request.form.get(
        "language",
        "en"
    )

    if language not in LANGUAGES:
        language = "en"

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return (
            t(
                language,
                "no_image"
            ),
            400
        )

    image_file = request.files[
        "image"
    ]

    if image_file.filename == "":

        return (
            t(
                language,
                "select_image"
            ),
            400
        )

    # --------------------------------------------------------
    # Create UNIQUE filename
    # --------------------------------------------------------

    original_filename = secure_filename(
        image_file.filename
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    if extension not in allowed_extensions:

        invalid_messages = {

            "en":
                "Please upload a JPG, PNG or WEBP image.",

            "hi":
                "कृपया JPG, PNG या WEBP तस्वीर अपलोड करें।",

            "mr":
                "कृपया JPG, PNG किंवा WEBP फोटो अपलोड करा."
        }

        return (
            invalid_messages[
                language
            ],
            400
        )

    filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        filename
    )

    image_file.save(
        filepath
    )

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    try:

        image = Image.open(
            filepath
        ).convert("RGB")

    except Exception:

        invalid_messages = {

            "en":
                "Invalid image file.",

            "hi":
                "अमान्य तस्वीर फ़ाइल।",

            "mr":
                "अवैध फोटो फाइल."
        }

        return (
            invalid_messages[
                language
            ],
            400
        )

    # --------------------------------------------------------
    # Analyze image
    # --------------------------------------------------------

    quality_problems = check_image_quality(
        image
    )

    result = analyze_image(
        image
    )

    top1 = result["top1"]
    top2 = result["top2"]
    top3 = result["top3"]

    confidence = result[
        "confidence"
    ]

    top2_confidence = result[
        "top2_confidence"
    ]

    top3_confidence = result[
        "top3_confidence"
    ]

    margin = result[
        "margin"
    ]

    agreement = result[
        "agreement"
    ]

    predicted_similarity = result[
        "predicted_similarity"
    ]

    nearest_similarity = result[
        "nearest_similarity"
    ]

    nearest_reference_index = result[
        "nearest_reference_index"
    ]

    # --------------------------------------------------------
    # Disease labels
    # --------------------------------------------------------

    disease = class_names[
        top1
    ]

    top2_name = translate_label(
        class_names[top2],
        language
    )

    top3_name = translate_label(
        class_names[top3],
        language
    )

    nearest_reference_name = translate_label(
        class_names[
            nearest_reference_index
        ],
        language
    )

    # --------------------------------------------------------
    # Reliability
    # --------------------------------------------------------

    reliability_reasons = []

    unknown_image = False

    if (
        predicted_similarity
        < similarity_threshold
    ):

        unknown_image = True

        reliability_reasons.append(
            make_reliability_reason(
                "The image is not sufficiently similar to the training examples for the predicted class.",
                language
            )
        )

    if (
        nearest_reference_index
        != top1
    ):

        unknown_image = True

        reliability_reasons.append(
            make_reliability_reason(
                "The classifier prediction and the closest learned class are different.",
                language
            )
        )

    if margin < 10:

        reliability_reasons.append(
            make_reliability_reason(
                "The top predictions are very close.",
                language
            )
        )

    if agreement < 2:

        reliability_reasons.append(
            make_reliability_reason(
                "The prediction changes when the image is viewed differently.",
                language
            )
        )

    for problem in quality_problems:

        reliability_reasons.append(
            make_reliability_reason(
                problem,
                language
            )
        )

    # --------------------------------------------------------
    # Crop Health Score
    # --------------------------------------------------------

    if unknown_image:

        health_score = None

        health_status = HEALTH_UI[
            language
        ]["unavailable"]

    else:

        health_score = calculate_health_score(
            disease,
            confidence,
            predicted_similarity,
            quality_problems
        )

        health_status = get_health_status(
            health_score,
            language
        )

    # --------------------------------------------------------
    # Confidence status
    # --------------------------------------------------------

    if unknown_image:

        confidence_status = "low"

        confidence_level = UI[
            language
        ]["ai_uncertain"]

        reliability_message = (
            UI[language]["uncertain_text"]
            + " "
            + UI[language]["do_not_rely"]
        )

    elif (
        confidence >= 75
        and margin >= 20
        and agreement >= 2
        and len(quality_problems) == 0
    ):

        confidence_status = "high"

        confidence_level = UI[
            language
        ]["high_confidence"]

        reliability_message = (
            RELIABILITY_UI[
                language
            ]["consistent"]
        )

    else:

        confidence_status = "medium"

        confidence_level = UI[
            language
        ]["moderate_confidence"]

        reliability_message = UI[
            language
        ]["clearer_image"]

    # --------------------------------------------------------
    # Disease information
    # --------------------------------------------------------

    from disease_data import disease_data

    if disease in disease_data:

        information = disease_data[
            disease
        ]

    else:

        parts = disease.split(
            "___"
        )

        crop_name = parts[0]

        disease_name = (
            parts[1].replace(
                "_",
                " "
            )
            if len(parts) > 1
            else disease
        )

        information = {

            "crop":
                crop_name,

            "disease":
                disease_name,

            "symptoms": [

                "The AI model detected this disease.",

                "Check the affected leaves for characteristic symptoms."
            ],

            "fertilizer": [

                "Maintain balanced plant nutrition.",

                "Use soil testing before applying corrective fertilizer."
            ],

            "prevention": [

                "Remove severely infected plant material.",

                "Maintain good air circulation.",

                "Avoid unnecessary leaf wetness.",

                "Monitor the crop regularly."
            ]
        }

    translated_information = translate_information(
        information,
        language
    )

    display_top1_name = (
        translated_information[
            "disease"
        ]
    )

    # --------------------------------------------------------
    # Display uncertain result
    # --------------------------------------------------------

    if confidence_status == "low":

        display_crop = "Uncertain"

        display_disease = UI[
            language
        ]["no_reliable_diagnosis"]

        if language == "hi":

            display_crop = "अनिश्चित"

        elif language == "mr":

            display_crop = "अनिश्चित"

    else:

        display_crop = translated_information[
            "crop"
        ]

        display_disease = translated_information[
            "disease"
        ]

    # --------------------------------------------------------
    # SAVE HISTORY
    #
    # IMPORTANT:
    # Store the ORIGINAL model labels in database.
    # Do not store translated Hindi/Marathi values.
    # --------------------------------------------------------

    history_crop = (
        disease.split(
            "___"
        )[0]
    )

    history_disease = (

        disease.split(
            "___",
            1
        )[1]

        if "___" in disease

        else disease
    )

    conn = sqlite3.connect(
        DATABASE_PATH
    )

    conn.execute(
        """
        INSERT INTO history
        (
            created_at,
            language,
            crop,
            disease,
            confidence,
            health_score
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),

            language,

            history_crop,

            history_disease,

            round(
                confidence,
                2
            ),

            health_score
        )
    )

    conn.commit()
    conn.close()

    # --------------------------------------------------------
    # RESULT PAGE
    # --------------------------------------------------------

    return render_template(

        "result.html",

        language=language,

        languages=LANGUAGES,

        ui=UI[language],

        image=filename,

        information=
            translated_information,

        display_crop=
            display_crop,

        display_disease=
            display_disease,

        top1_name=
            display_top1_name,

        confidence=
            round(
                confidence,
                2
            ),

        top2_confidence=
            round(
                top2_confidence,
                2
            ),

        top3_confidence=
            round(
                top3_confidence,
                2
            ),

        margin=
            round(
                margin,
                2
            ),

        agreement=
            agreement,

        predicted_similarity=
            round(
                predicted_similarity,
                4
            ),

        nearest_similarity=
            round(
                nearest_similarity,
                4
            ),

        similarity_threshold=
            round(
                similarity_threshold,
                4
            ),

        nearest_reference_name=
            nearest_reference_name,

        top2_name=
            top2_name,

        top3_name=
            top3_name,

        confidence_level=
            confidence_level,

        confidence_status=
            confidence_status,

        reliability_message=
            reliability_message,

        reliability_reasons=
            reliability_reasons,

        health_score=
            health_score,

        health_status=
            health_status,

        health_ui=
            HEALTH_UI[language],

        history_url=
            url_for(
                "history",
                lang=language
            ),

        map_url=
            url_for(
                "outbreak_map",
                lang=language
            ),

        report_url=
            url_for(
                "report_outbreak"
            ),

        raw_disease=
            disease,

        raw_crop=
            disease.split(
                "___"
            )[0]
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )