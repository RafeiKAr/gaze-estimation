# Bauanleitung für den Hugging Face Space (SDK: Docker)
FROM python:3.11-slim

# Hugging Face verlangt einen normalen Benutzer mit der ID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /app

# zuerst nur die Pakete installieren (wird zwischengespeichert, Neubau geht schneller)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# dann den Code und das Modell kopieren
COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
