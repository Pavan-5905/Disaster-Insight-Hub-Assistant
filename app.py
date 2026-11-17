from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, url_for
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spacy
except ImportError:  # spaCy is optional; Groq can still extract a location.
    spacy = None


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True

KB_PATH = BASE_DIR / os.getenv("KB_FILE", "safety_guidelines.xlsx")
CRISIS_DATA_PATH = BASE_DIR / os.getenv("CRISIS_DATA_FILE", "crisis.csv")
MODEL_PATH = BASE_DIR / os.getenv("CLASSIFIER_FILE", "random_forest_model.joblib")
VECTORIZER_PATH = BASE_DIR / os.getenv("VECTORIZER_FILE", "tfidf_vectorizer.joblib")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")

SYSTEM_PROMPT = """You are the Disaster Insight Hub Assistant (DIHA).
Give calm, clear, practical disaster-safety guidance.

Safety and accuracy rules:
- Treat the supplied safety knowledge as the primary source for advice.
- Do not invent evacuation routes, shelters, phone numbers, weather conditions,
  official alerts, or instructions that are not present in the supplied context.
- If the user appears to be in immediate danger, advise contacting local
  emergency services and following instructions from local authorities.
- Give 2 to 4 concise, actionable steps. Keep the response under 150 words.
- Do not claim that you have contacted emergency services or verified live conditions.
- Extract a location only when it is explicitly present in the user's original
  message. Copy the exact location phrase from that message. Never infer a
  location from the crisis type, language, or model context.
- If no explicit place is mentioned, return null for extracted_location.
"""

LOCATION_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "extracted_location": {"type": ["string", "null"]},
    },
    "required": ["assistant_reply", "extracted_location"],
    "additionalProperties": False,
}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("diha")


# -----------------------------------------------------------------------------
# Model and data loading
# -----------------------------------------------------------------------------
def load_landing_metrics() -> Dict[str, Any]:
    """Load lightweight dataset metrics for the landing page, if available."""
    try:
        df = pd.read_csv(CRISIS_DATA_PATH)
    except Exception as exc:
        logger.warning("Could not load crisis dataset metrics: %s", exc)
        return {}

    metrics: Dict[str, Any] = {
        "records": int(len(df)),
        "crisis_types": int(df["crisis_type"].nunique()) if "crisis_type" in df else 0,
        "locations": int(df["location"].nunique()) if "location" in df else 0,
        "top_locations": [],
    }

    if "location" in df.columns:
        metrics["top_locations"] = df["location"].dropna().value_counts().head(3).index.tolist()

    return metrics


def safe_joblib_load(path: Path) -> Optional[Any]:
    try:
        if not path.is_file():
            logger.error("Required model artifact not found: %s", path.name)
            return None
        return joblib.load(path)
    except Exception as exc:
        logger.exception("Could not load model artifact %s: %s", path.name, exc)
        return None


def load_knowledge_base(path: Path) -> pd.DataFrame:
    """Load and normalize the Excel safety knowledge base."""
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        logger.warning("Could not load knowledge base %s: %s", path.name, exc)
        return pd.DataFrame(columns=["crisis_type", "title", "advice", "doc_text"])

    required_cols = {"crisis_type", "title", "advice"}
    missing = required_cols.difference(df.columns)
    if missing:
        logger.error("Knowledge base is missing required columns: %s", sorted(missing))
        return pd.DataFrame(columns=["crisis_type", "title", "advice", "doc_text"])

    for col in ["crisis_type", "title", "advice"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df.loc[df["crisis_type"] == "", "crisis_type"] = "General"
    df["doc_text"] = (df["title"] + " " + df["advice"]).str.strip()
    df = df[df["doc_text"].str.len() > 0].reset_index(drop=True)
    return df


def build_retriever(df: pd.DataFrame):
    if df.empty:
        return None, None

    from sklearn.feature_extraction.text import TfidfVectorizer

    kb_vectorizer = TfidfVectorizer(stop_words="english")
    kb_matrix = kb_vectorizer.fit_transform(df["doc_text"])
    return kb_vectorizer, kb_matrix


landing_metrics = load_landing_metrics()
best_model = safe_joblib_load(MODEL_PATH)
classifier_vectorizer = safe_joblib_load(VECTORIZER_PATH)
kb_df = load_knowledge_base(KB_PATH)
kb_vectorizer, kb_matrix = build_retriever(kb_df)

nlp = None
if spacy is not None:
    try:
        nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_sm"))
        logger.info("spaCy location model loaded.")
    except Exception as exc:
        logger.warning("spaCy model unavailable; Groq location extraction remains enabled: %s", exc)


# -----------------------------------------------------------------------------
# Text processing and classification
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Keep preprocessing consistent with the training pipeline."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_location_spacy(text: str) -> Optional[str]:
    """Return the first spaCy GPE/LOC mention as a local fallback."""
    if nlp is None:
        return None

    try:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC"}:
                candidate = ent.text.strip()
                if candidate:
                    return candidate
    except Exception as exc:
        logger.warning("spaCy location extraction failed: %s", exc)

    return None


def run_prediction(text_input: str) -> Dict[str, Any]:
    if best_model is None or classifier_vectorizer is None:
        return {"error": "Classifier artifacts are missing or could not be loaded."}

    try:
        cleaned_text = clean_text(text_input)
        text_tfidf = classifier_vectorizer.transform([cleaned_text])
        prediction = best_model.predict(text_tfidf)[0]

        # The deployed Random Forest predicts crisis_type and location in its
        # training script, but this application deliberately uses spaCy as the
        # location fallback, consistent with the original app.py pipeline.
        crisis_type = str(prediction[0])
        location = extract_location_spacy(text_input) or "Unknown"

        return {
            "original_text": text_input,
            "predicted_crisis_type": crisis_type,
            "predicted_location": location,
        }
    except Exception as exc:
        logger.exception("Crisis prediction failed: %s", exc)
        return {"error": "The crisis classification step failed."}


# -----------------------------------------------------------------------------
# Retrieval-Augmented Generation
# -----------------------------------------------------------------------------
def retrieve_guidance(
    crisis_type: str,
    query: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    if kb_df.empty or kb_vectorizer is None or kb_matrix is None:
        return []

    try:
        query_vec = kb_vectorizer.transform([f"{crisis_type} {query}"])
        mask = kb_df["crisis_type"].str.casefold() == crisis_type.casefold()

        if not bool(mask.any()):
            mask = pd.Series(True, index=kb_df.index)

        filtered_df = kb_df.loc[mask].reset_index(drop=True)
        matrix_slice = kb_matrix[mask.to_numpy()]
        similarities = cosine_similarity(query_vec, matrix_slice)[0]
        top_indices = similarities.argsort()[::-1][: max(1, top_k)]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            row = filtered_df.iloc[int(idx)]
            results.append(
                {
                    "crisis_type": row["crisis_type"],
                    "title": row["title"],
                    "advice": row["advice"],
                    "similarity": float(similarities[int(idx)]),
                }
            )
        return results
    except Exception as exc:
        logger.exception("Safety-guidance retrieval failed: %s", exc)
        return []


def build_assistant_messages(
    user_query: str,
    crisis_type: str,
    guidance: List[Dict[str, Any]],
    location_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    if guidance:
        context_lines = [
            f"{index + 1}. {item['title']}: {item['advice']}"
            for index, item in enumerate(guidance)
        ]
        context_block = "\n".join(context_lines)
    else:
        context_block = "No verified safety guidance was retrieved from the knowledge base."

    known_location = location_hint if location_hint and location_hint != "Unknown" else "Not identified by spaCy"

    user_content = f"""Original user message:
{user_query}

Detected crisis type:
{crisis_type}

Location candidate from spaCy (may be unavailable; verify against the original message):
{known_location}

Retrieved safety knowledge:
{context_block}

Tasks:
1. Write a concise, helpful disaster-safety response using the retrieved knowledge.
2. Extract the location explicitly mentioned in the original user message.
3. For extracted_location, copy the exact location phrase from the original message.
4. If the original message contains no explicit geographical location, use null.
5. Never treat the spaCy candidate as correct unless it appears in the original message.

Return a human-readable assistant_reply of no more than 150 words.
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# -----------------------------------------------------------------------------
# Groq client and response generation
# -----------------------------------------------------------------------------
def get_groq_client() -> Tuple[Optional[Groq], Optional[str]]:
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key:
        return None, "GROQ_API_KEY is not configured."

    try:
        return Groq(api_key=api_key), None
    except Exception as exc:
        logger.exception("Could not initialize Groq client: %s", exc)
        return None, "Could not initialize the Groq client."


def normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def validate_extracted_location(candidate: Any, original_query: str) -> Optional[str]:
    """Accept only a location phrase that is literally present in the user's query."""
    if not isinstance(candidate, str):
        return None

    candidate = candidate.strip()
    if not candidate:
        return None

    if normalize_for_match(candidate) not in normalize_for_match(original_query):
        logger.warning("Rejected model-extracted location not present in the original query.")
        return None

    return candidate


def call_groq_assistant(
    user_query: str,
    crisis_type: str,
    guidance: List[Dict[str, Any]],
    location_hint: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Generate the answer and extract the explicit location in one Groq call."""
    client, client_error = get_groq_client()
    if client_error or client is None:
        return None, None, client_error

    messages = build_assistant_messages(
        user_query=user_query,
        crisis_type=crisis_type,
        guidance=guidance,
        location_hint=location_hint,
    )

    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            reasoning_effort=GROQ_REASONING_EFFORT,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diha_assistant_response",
                    "strict": True,
                    "schema": LOCATION_REPLY_SCHEMA,
                },
            },
        )

        message = completion.choices[0].message
        content = message.content
        if not content:
            return None, None, "Groq returned an empty response."

        result = json.loads(content)
        assistant_reply = result.get("assistant_reply")
        if not isinstance(assistant_reply, str) or not assistant_reply.strip():
            return None, None, "Groq did not return a valid assistant reply."

        extracted_location = validate_extracted_location(
            result.get("extracted_location"), user_query
        )

        return assistant_reply.strip(), extracted_location, None

    except json.JSONDecodeError:
        logger.exception("Groq returned content that could not be parsed as JSON.")
        return None, None, "Groq returned an invalid structured response."
    except Exception as exc:
        # Do not expose API keys or request headers in the client response.
        logger.exception("Groq request failed: %s", exc)
        status_code = getattr(exc, "status_code", None)
        if status_code == 401:
            return None, None, "Groq authentication failed. Check GROQ_API_KEY."
        if status_code == 402:
            return None, None, "Groq account requires billing or usage credits."
        if status_code == 429:
            return None, None, "Groq rate limit reached. Try again later."
        return None, None, "Groq response generation failed. Check server logs."


# -----------------------------------------------------------------------------
# Safe fallback
# -----------------------------------------------------------------------------
def build_fallback_reply(crisis_type: str, guidance: List[Dict[str, Any]]) -> str:
    if not guidance:
        return (
            f"A {crisis_type.lower()} scenario was detected, but the AI response service "
            "is unavailable and no matching verified guidance was retrieved. "
            "Follow instructions from local authorities, move away from immediate danger "
            "when safe to do so, and contact local emergency services if anyone is at risk."
        )

    highlights = " ".join(
        f"{item['title']}: {item['advice']}" for item in guidance
    )
    return (
        f"Verified guidance for the detected {crisis_type.lower()} situation: "
        f"{highlights} Contact local emergency services if you are in immediate danger."
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    model_ready = best_model is not None and classifier_vectorizer is not None
    kb_ready = not kb_df.empty
    groq_ready = bool(os.getenv("GROQ_API_KEY"))
    assistant_status = (
        "Online & ready" if model_ready and kb_ready and groq_ready else "Degraded mode"
    )

    return render_template(
        "index.html",
        metrics=landing_metrics,
        hero_status=assistant_status,
        predict_url=url_for("predict"),
        assistant_url=url_for("assistant"),
        kb_docs=len(kb_df),
    )


@app.route("/health", methods=["GET"])
def health():
    """Basic readiness endpoint; never returns secret values."""
    return jsonify(
        {
            "status": "ok",
            "classifier_ready": best_model is not None and classifier_vectorizer is not None,
            "knowledge_base_ready": not kb_df.empty,
            "groq_configured": bool(os.getenv("GROQ_API_KEY")),
            "groq_model": GROQ_MODEL,
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    text_input = payload.get("text", "")

    if not isinstance(text_input, str) or not text_input.strip():
        return jsonify({"error": "No text provided for prediction."}), 400

    result = run_prediction(text_input.strip())
    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)


@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json(silent=True) if request.is_json else request.form
    payload = payload or {}
    user_query = payload.get("query") or payload.get("text") or ""

    if not isinstance(user_query, str) or not user_query.strip():
        return jsonify({"error": "No prompt provided."}), 400

    user_query = user_query.strip()
    classification = run_prediction(user_query)
    if "error" in classification:
        return jsonify(classification), 500

    crisis_type = classification["predicted_crisis_type"]
    spacy_location = classification.get("predicted_location")
    guidance = retrieve_guidance(crisis_type, user_query, top_k=3)

    llm_reply, groq_location, llm_error = call_groq_assistant(
        user_query=user_query,
        crisis_type=crisis_type,
        guidance=guidance,
        location_hint=spacy_location,
    )

    # Prefer the validated Groq extraction; otherwise retain spaCy's local result.
    final_location = groq_location or spacy_location or "Unknown"

    if not llm_reply:
        llm_reply = build_fallback_reply(crisis_type, guidance)

    return jsonify(
        {
            "crisis_type": crisis_type,
            "location_hint": final_location,
            "guidance": guidance,
            "assistant_reply": llm_reply,
            "llm_error": llm_error,
        }
    )


if __name__ == "__main__":
    logger.info("Starting Disaster Insight Hub Assistant")
    logger.info("Groq model configured: %s", GROQ_MODEL)
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5001")))