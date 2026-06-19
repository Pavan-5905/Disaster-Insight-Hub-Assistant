from flask import Flask, request, jsonify, render_template, url_for
import joblib
import pandas as pd
import re
import spacy
import os
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ollama import Client

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True

KB_PATH = "safety_guidelines.xlsx"
OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")
SYSTEM_PROMPT = (
    "You are the Disaster Insight Hub Assistant. "
    "Respond with calm, empathetic safety guidance tailored to the detected crisis type. "
    "Use only the supplied safety knowledge. "
    "Offer 2-4 concise actionable steps, mention escalation points, and avoid speculation."
)


# -----------------------------------------------------------------------------
# Model + data bootstrapping
# -----------------------------------------------------------------------------
def load_landing_metrics() -> Dict[str, Any]:
    """Collect lightweight dataset insights for the landing page."""
    try:
        df = pd.read_csv("crisis.csv")
    except Exception as exc:
        print(f"Could not load crisis.csv for landing metrics: {exc}")
        return {}

    metrics = {
        "records": len(df),
        "crisis_types": df["crisis_type"].nunique() if "crisis_type" in df else 0,
        "locations": df["location"].nunique() if "location" in df else 0,
        "top_locations": [],
    }

    if "location" in df:
        metrics["top_locations"] = df["location"].value_counts().head(3).index.tolist()

    return metrics


def safe_joblib_load(path: str) -> Optional[Any]:
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"Error loading {path}: {exc}")
        return None


def load_knowledge_base(path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(path)
    except Exception as exc:
        print(f"Could not load knowledge base at {path}: {exc}")
        return pd.DataFrame()

    required_cols = {"crisis_type", "title", "advice"}
    if not required_cols.issubset(df.columns):
        print(f"Knowledge base missing required columns: {required_cols}")
        return pd.DataFrame()

    df = df.fillna({"title": "", "advice": "", "crisis_type": "General"})
    df["doc_text"] = (df["title"].str.strip() + " " + df["advice"].str.strip()).str.strip()
    df = df[df["doc_text"].str.len() > 0].reset_index(drop=True)
    return df


def build_retriever(df: pd.DataFrame):
    if df.empty:
        return None, None

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["doc_text"])
    return vectorizer, matrix


landing_metrics = load_landing_metrics()
best_model = safe_joblib_load("random_forest_model.joblib")
vectorizer = safe_joblib_load("tfidf_vectorizer.joblib")
kb_df = load_knowledge_base(KB_PATH)
kb_vectorizer, kb_matrix = build_retriever(kb_df)

try:
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded successfully.")
except Exception as exc:
    print(f"Error loading spaCy model: {exc}")
    nlp = None


# -----------------------------------------------------------------------------
# Text + retrieval helpers
# -----------------------------------------------------------------------------
def clean_text(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"http\S+", "", t)      # remove urls
    t = re.sub(r"@\w+", "", t)         # remove mentions
    t = re.sub(r"#", "", t)            # remove hashtags symbol
    t = re.sub(r"[^a-zA-Z\s]", "", t)  # keep only letters
    t = re.sub(r"\s+", " ", t).strip()
    return t


def extract_location(text: str) -> Optional[str]:
    if not nlp:
        return None

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:  # GPE = Countries, Cities, States
            return ent.text
    return None


def run_prediction(text_input: str) -> Dict[str, str]:
    if not (best_model and vectorizer):
        return {
            "error": "Model artifacts missing. Retrain or copy the .joblib files."
        }

    cleaned_text = clean_text(text_input)
    text_tfidf = vectorizer.transform([cleaned_text])
    pred = best_model.predict(text_tfidf)[0]
    crisis_pred = pred[0]
    ner_location = extract_location(text_input) if nlp else None
    final_location = ner_location if ner_location else "Unknown"

    return {
        "original_text": text_input,
        "predicted_crisis_type": crisis_pred,
        "predicted_location": final_location,
    }


def retrieve_guidance(crisis_type: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    if kb_df.empty or kb_vectorizer is None or kb_matrix is None:
        return []

    query_text = f"{crisis_type} {query}"
    query_vec = kb_vectorizer.transform([query_text])

    mask = kb_df["crisis_type"].str.lower() == crisis_type.lower()
    if mask.sum() == 0:
        mask = pd.Series([True] * len(kb_df), index=kb_df.index)

    matrix_slice = kb_matrix[mask.values]
    similarities = cosine_similarity(query_vec, matrix_slice)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    filtered_df = kb_df[mask].reset_index(drop=True)
    guidance = []
    for idx in top_indices:
        row = filtered_df.iloc[idx]
        guidance.append(
            {
                "crisis_type": row["crisis_type"],
                "title": row["title"],
                "advice": row["advice"],
                "similarity": float(similarities[idx]),
            }
        )
    return guidance


def build_assistant_messages(user_query: str, crisis_type: str, guidance: List[Dict[str, Any]]):
    if guidance:
        context_lines = [
            f"{i+1}. {item['title']}: {item['advice']}"
            for i, item in enumerate(guidance)
        ]
        context_block = "\n".join(context_lines)
    else:
        context_block = "No verified safety guidance was retrieved."

    user_content = (
        f"User query: {user_query}\n"
        f"Detected crisis type: {crisis_type}\n"
        f"Safety knowledge:\n{context_block}\n\n"
        "Return a short, human response (≤150 words)."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def get_ollama_client():
    api_key = os.environ.get("OLLAMA_API_KEY","d3f0ddc5645742cbb0b3f4fe30d0fce5.ZZrTcltDam1bLmdJ2JtOPBVb")
    if not api_key:
        return None, "OLLAMA_API_KEY missing."
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        client = Client(host=OLLAMA_HOST, headers=headers)
        return client, None
    except Exception as exc:
        print(f"Failed to initialize Ollama client: {exc}")
        return None, str(exc)


def call_ollama_assistant(user_query: str, crisis_type: str, guidance: List[Dict[str, Any]]):
    client, err = get_ollama_client()
    if err:
        return None, err

    messages = build_assistant_messages(user_query, crisis_type, guidance)

    try:
        parts = []
        for chunk in client.chat(OLLAMA_MODEL, messages=messages, stream=True):
            content = chunk.get("message", {}).get("content")
            if content:
                parts.append(content)
        reply = "".join(parts).strip()
        return (reply or None), None
    except Exception as exc:
        print(f"Ollama request failed: {exc}")
        return None, str(exc)


def build_fallback_reply(crisis_type: str, guidance: List[Dict[str, Any]]):
    if not guidance:
        return (
            f"I detected a {crisis_type.lower()} scenario but couldn't reach the language model. "
            "Follow local authority instructions, keep emergency contacts accessible, and move to safety."
        )

    highlights = "; ".join(f"{item['title']}: {item['advice']}" for item in guidance)
    return (
        f"Here is verified guidance for a {crisis_type.lower()} situation: {highlights}. "
        "Contact emergency services if you are in immediate danger."
    )


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    model_ready = all([best_model, vectorizer])
    kb_ready = not kb_df.empty
    assistant_status = "Online & ready" if model_ready and kb_ready else "Degraded mode"

    return render_template(
        "index.html",
        metrics=landing_metrics,
        hero_status=assistant_status,
        predict_url=url_for("predict"),
        assistant_url=url_for("assistant"),
        kb_docs=len(kb_df),
    )


@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        text_input = request.get_json(silent=True) or {}
        text_input = text_input.get("text", "")
    else:
        text_input = request.form.get("text", "")

    if not text_input:
        return jsonify({"error": "No text provided for prediction."}), 400

    payload = run_prediction(text_input)
    if "error" in payload:
        return jsonify(payload), 500

    return jsonify(payload)


@app.route("/assistant", methods=["POST"])
def assistant():
    payload = request.get_json(silent=True) or request.form
    user_query = payload.get("query") or payload.get("text") or ""

    if not user_query:
        return jsonify({"error": "No prompt provided."}), 400

    classification = run_prediction(user_query)
    if "error" in classification:
        return jsonify(classification), 500

    crisis_type = classification["predicted_crisis_type"]
    guidance = retrieve_guidance(crisis_type, user_query)
    llm_reply, llm_error = call_ollama_assistant(user_query, crisis_type, guidance)

    if not llm_reply:
        llm_reply = build_fallback_reply(crisis_type, guidance)

    return jsonify(
        {
            "crisis_type": crisis_type,
            "location_hint": classification["predicted_location"],
            "guidance": guidance,
            "assistant_reply": llm_reply,
            "llm_error": llm_error,
        }
    )


if __name__ == "__main__":
    print("\n--- Disaster Insight Hub Assistant ---")
    print("Open http://127.0.0.1:5001 in your browser after running `python app.py`.")
    app.run(debug=False, host="0.0.0.0", port=5001)