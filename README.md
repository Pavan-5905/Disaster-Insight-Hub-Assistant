# Disaster Insight Hub Assistant

Flask-based Retrieval-Augmented Generation (RAG) experience that classifies crisis-related text, fetches safety playbooks, and drafts natural responses through Groq using `openai/gpt-oss-120b`. The Random Forest classifier handles crisis detection, while TF-IDF retrieval grounds the assistant in your curated Excel knowledge base.

## Contents
- `disaster.py` / `disaster.ipynb`: original notebook-to-script pipeline for cleaning text, training multiple models, comparing scores, and exporting the best estimator + vectorizer.
- `random_forest_model.joblib`, `tfidf_vectorizer.joblib`: serialized artifacts consumed by the web server.
- `app.py`: Flask application exposing the landing page, `/health`, `/predict`, and `/assistant` APIs.
- `templates/index.html`, `static/css/styles.css`: landing page UI assets.
- `crisis.csv`: training/evaluation dataset containing `text`, `crisis_type`, `location`, etc.

## Requirements
- Python >= 3.9
- `pip install -r requirements.txt` (or install individually):
  - `flask`
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `spacy`
  - `openpyxl` (Excel reader)
-  `groq`

### Optional (for NER-powered location extraction)
```
python -m spacy download en_core_web_sm
```

### Environment variables
| Name | Purpose |
| --- | --- |
| `GROQ_API_KEY` | Required to authenticate requests to Groq. |
| `GROQ_MODEL` | Optional. Defaults to `openai/gpt-oss-120b`. |
| `GROQ_REASONING_EFFORT` | Optional. Defaults to `low`. |
| `SPACY_MODEL` | Optional. spaCy model name; defaults to `en_core_web_sm`. |

## Running Locally
1. Ensure the serialized artifacts (`random_forest_model.joblib`, `tfidf_vectorizer.joblib`) and `safety_guidelines.xlsx` exist at the project root. Re-run `disaster.py` if you need fresh models, and edit the Excel file to update guidance.
2. Set the Groq configuration (copy `.env.example` to `.env` if using a dotenv
   loader, or export the variables in your shell):
   ```
   export GROQ_API_KEY="your_groq_api_key"
   export GROQ_MODEL="openai/gpt-oss-120b"
   export GROQ_REASONING_EFFORT="low"
   ```
3. Start the Flask server:
   ```
   python app.py
   ```
4. Browse to `http://127.0.0.1:5001`:
   - The landing page shows pipeline status + stats.
   - The “Live RAG Chat” widget posts to `/assistant` for natural replies.

## API Usage
- `POST /assistant`
  - Body: `{ "query": "What should I do during a cyclone?" }`
  - Response:
    ```json
    {
      "crisis_type": "Cyclone",
      "location_hint": "Unknown",
      "guidance": [
        {
          "crisis_type": "Cyclone",
          "title": "Secure shelter",
          "advice": "Stay indoors...",
          "similarity": 0.71
        }
      ],
      "assistant_reply": "Stay indoors ...",
      "llm_error": null
    }
    ```
- `POST /predict` (legacy classifier endpoint) still accepts `{ "text": "..." }` and returns the raw model output (useful for debugging).
- `GET /health` reports classifier, knowledge-base, and Groq configuration readiness
  without returning secrets.

## Landing Page Features
- Hero section with operational stats and health badge.
- Pipeline cards describing Detect → Retrieve → Generate flow.
- Live chat interface that streams user/assistant bubbles plus retrieved playbooks.
- Insight side panel showing predicted crisis type, location hint, and the TF-IDF matches.

## Extending the Project
- **Model experimentation**: adjust `disaster.py` to fine-tune classifiers, export updated `.joblib` artifacts, and restart the web stack.
- **Knowledge expansion**: edit `safety_guidelines.xlsx` with more crisis types or sources—no code changes required for ingestion.
- **Observability**: log `/assistant` requests, persist chat history, or integrate tracing (OpenTelemetry) for production readiness.
- **Deployment**: containerize the Flask app, add HTTPS + authentication, and run behind a WAF/load balancer.

## Troubleshooting
- `Model artifacts missing`: confirm the `.joblib` files exist and match the training schema.
- `Knowledge base missing`: ensure `safety_guidelines.xlsx` is present and has `crisis_type`, `title`, `advice` columns.
- `GROQ_API_KEY missing`: export the key; the application remains available with
  template-based fallback replies.
- `en_core_web_sm` missing: install via spaCy command; without it, location hints default to `"Unknown"`.
- `Groq response generation failed`: confirm the API key, model name, account
  access, and rate limits. The application logs the server-side failure and
  returns verified knowledge-base guidance as a fallback.

## Deployment

### Docker

Build and run the included image:

```bash
docker build -t diha .
docker run --rm -p 5001:5001 -e GROQ_API_KEY="$GROQ_API_KEY" diha
```

### Vercel

Vercel can detect the Flask entry point in `app.py`. Configure `GROQ_API_KEY`
and, if needed, `GROQ_MODEL` and `GROQ_REASONING_EFFORT` as Vercel project
environment variables. Do not commit `.env` files or API keys. The serialized
model artifacts and Excel knowledge base must be included in the deployment.
