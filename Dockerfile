# ---- Base image ----
FROM python:3.10-slim


# ---- Working directory ----
WORKDIR /app

# ---- Copy requirements first (better layer caching) ----
COPY requirements.txt .

# ---- Install Python dependencies ----
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---- Install spaCy language model ----
RUN python -m spacy download en_core_web_sm

# ---- Copy application code ----
COPY . .

# ---- Expose Flask port ----
EXPOSE 5001

# ---- Run the app ----
CMD ["python", "app.py"]
