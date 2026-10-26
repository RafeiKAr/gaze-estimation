"""Server der Eye-Tracking-Demo (FastAPI).

Starten (lokal):   uvicorn app:app --port 7860
Dann im Browser:   http://localhost:7860

Schnittstellen (alle POST mit JSON, Bilder als JPEG-Data-URL):
  /api/session          neue Sitzung anlegen          -> {"sid": ...}
  /api/calib_frame      ein Kalibrierbild + Punkt      -> {"ok": true, "used": ...}
  /api/calibrate        Kopf trainieren                -> Kennzahlen (Fehler vorher/nachher)
  /api/predict          ein Bild -> Blickpunkt         -> {"base": [x,y], "personal": [x,y]}
  /api/end              Sitzung löschen
  GET /api/health       Prüfen, ob der Server läuft

Datenschutz: Bilder werden nur im Arbeitsspeicher in Merkmale umgerechnet und
nicht gespeichert. Sitzungen werden nach 30 Minuten ohne Nutzung gelöscht.
"""

import base64
import io
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from gaze_model import GazeModel, Session

STATIC = Path(__file__).parent / "static"
SESSION_TIMEOUT = 30 * 60          # Sekunden
MAX_SESSIONS = 20

app = FastAPI(title="Gaze Personalization Demo")


@app.middleware("http")
async def no_cache(request, call_next):
    """Browser soll immer die neueste Version von Seite und Skript laden."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store"
    return response


app.mount("/static", StaticFiles(directory=STATIC), name="static")
gaze_model = GazeModel()           # Basismodell einmal laden
sessions: dict[str, Session] = {}
lock = threading.Lock()


class SidIn(BaseModel):
    sid: str


class CalibFrameIn(BaseModel):
    sid: str
    image: str                     # "data:image/jpeg;base64,..."
    x: float                       # Punktposition, normiert 0..1
    y: float
    t: float                       # Sekunden seit Start der Kalibrierung


class PredictIn(BaseModel):
    sid: str
    image: str


def decode_image(data_url: str) -> Image.Image:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bild konnte nicht gelesen werden: {exc}")


def get_session(sid: str) -> Session:
    s = sessions.get(sid)
    if s is None:
        raise HTTPException(404, "Sitzung unbekannt oder abgelaufen. Bitte Seite neu laden.")
    return s


def cleanup():
    now = time.time()
    with lock:
        for sid in [k for k, s in sessions.items() if now - s.last_used > SESSION_TIMEOUT]:
            del sessions[sid]


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"ok": True, "sessions": len(sessions)}


@app.post("/api/session")
def new_session():
    cleanup()
    with lock:
        if len(sessions) >= MAX_SESSIONS:
            oldest = min(sessions, key=lambda k: sessions[k].last_used)
            del sessions[oldest]
        sid = uuid.uuid4().hex
        sessions[sid] = Session(gaze_model)
    return {"sid": sid}


@app.post("/api/calib_frame")
def calib_frame(d: CalibFrameIn):
    s = get_session(d.sid)
    used = s.add_calibration_frame(decode_image(d.image), d.x, d.y, d.t)
    return {"ok": True, "used": used, "count": len(s.feats)}


@app.post("/api/calibrate")
def calibrate(d: SidIn):
    s = get_session(d.sid)
    try:
        return s.calibrate()
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.post("/api/predict")
def predict(d: PredictIn):
    s = get_session(d.sid)
    base, personal = s.predict(decode_image(d.image))
    return {"base": base, "personal": personal, "error": s.error_after}


class DebugIn(BaseModel):
    sid: str
    data: dict


@app.post("/api/debug_log")
def debug_log(d: DebugIn):
    """Nur mit GAZE_DEBUG=1: Test- und Verfolgungsdaten für die Fehlersuche speichern."""
    from gaze_model import DEBUG_DIR
    if DEBUG_DIR is not None:
        import json
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / "tracking_log.json").write_text(json.dumps(d.data), encoding="utf-8")
    return {"ok": True}


@app.post("/api/end")
def end(d: SidIn):
    with lock:
        sessions.pop(d.sid, None)
    return {"ok": True}
