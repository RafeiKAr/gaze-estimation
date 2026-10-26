"""Modell-Teil der Demo: Basismodell laden, Merkmale berechnen, Kopf personalisieren.

Grundlage ist das Verfahren der Masterarbeit (personal_exp.py, Kapitel 4):
  * ResNet-18 + Regressionskopf 512-512-256-128-2 mit Sigmoid (model.py)
  * Basismodell models/base_norm_subject.pt (train_base.py, Experiment E0)
  * Backbone eingefroren -> jedes Bild wird EINMAL in 512 Merkmale umgerechnet
  * nur der Kopf wird trainiert: AdamW, lr 1e-4, weight decay 1e-5, Batch 32,
    L1-Verlust, Early Stopping
  * Fehlermaß = Diagonalfehler E_diag in % der Bildschirmdiagonale (metrics.py)

Unterschiede zu den Experimenten (für die Live-Webcam nötig):
  * Bildausschnitt (GAZE_CROP): Bei der Webcam sitzt man weiter weg als bei den
    eigenen Aufnahmen, der Kopf bewegt sich kaum, und die Augen sind im ganzen
    Bild nur wenige Pixel groß. Deshalb wird ein Ausschnitt verwendet, dessen
    Größe beim ersten Gesicht festgelegt wird und der dem Gesicht in jedem Bild folgt:
        "eyes" (Standard): Streifen um beide Augen, vergrößert
        "face": Quadrat um das Gesicht
        "none": ganzes Bild (wie in der Masterarbeit)
    Das Basismodell ("vorher") bekommt immer das GANZE Bild, wie beim Training.
  * Validierung in zeitlichen BLÖCKEN: 10 Blöcke (je ca. 6 s), 2 davon dienen der
    Prüfung, mit einer Lücke von 3 Bildern zum Training.
  * Patience 15 statt 3 und mindestens 40 Epochen (nur ca. 8 Schritte pro Epoche,
    unruhige Prüfung mit nur ca. 60 Bildern).
  * Zusätzlich zwei lineare Ridge-Regressionen (wie die SVR-Kalibrierung bei
    iTracker): auf den 512 globalen Merkmalen und auf RÄUMLICHEN Merkmalen
    (layer3-Karte, auf ein kleines Raster gemittelt). Die globalen Merkmale sind
    ein Mittelwert über das ganze Bild und verlieren dadurch, WO die Iris liegt;
    die räumlichen Merkmale behalten diese Information. Verwendet wird das
    Verfahren mit dem kleinsten Prüffehler; alle Werte stehen im Ergebnis.

Umgebungsvariablen:
  GAZE_CROP=eyes|face|none   Bildausschnitt (Standard eyes)
  GAZE_DEBUG=1               jedes 30. Kalibrierbild (Ausschnitt) und einen Bericht
                             calib_report.json in debug_frames/ speichern (nur lokal!)
"""

import copy
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

MODEL_FILE = Path(__file__).parent / "models" / "base_norm_subject.pt"
DEBUG_DIR = Path(__file__).parent / "debug_frames" if os.environ.get("GAZE_DEBUG") == "1" else None
CROP_MODE = os.environ.get("GAZE_CROP", "eyes").lower()
if os.environ.get("GAZE_FACE_CROP") == "0":           # alte Einstellung weiter erlauben
    CROP_MODE = "none"
BASE_EVERY = 3            # Basismodell ("vorher") nur für jedes 3. Bild rechnen (spart Zeit)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

# Gleiche Bildvorverarbeitung wie beim Training (eval_tf in personal_exp.py)
# Augenstreifen: ohne Verzerrung auf 224 x 112 (Breite x Höhe) bringen
STRIP_TF = transforms.Compose([
    transforms.Resize((112, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# Räumliche Merkmale: layer3-Karte auf ein kleines Raster mitteln. Anders als die
# globalen 512 Merkmale (Mittelwert über das ganze Bild) bleibt hier erhalten, WO
# im Bild etwas ist – also auch, wo die Iris im Auge liegt.
SPATIAL_POOL = {"eyes": (3, 8), "face": (4, 4), "none": (4, 4)}
EVAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Trainingseinstellungen
LR, WEIGHT_DECAY, BATCH, PATIENCE, MAX_EPOCHS = 1e-4, 1e-5, 32, 15, 400
MIN_EPOCHS = 40           # vorher kein Abbruch: die Prüfung ist mit 60 Bildern sehr unruhig
RIDGE_FACTORS = (1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0)   # Regularisierung relativ zur Merkmalsgröße
SKIP_START_SEC = 0.5      # erste 0,5 s verwerfen (Blick ist noch nicht auf dem Punkt)
N_BLOCKS = 10             # Kalibrierung in 10 zeitliche Blöcke teilen ...
VAL_EVERY = 5             # ... jeder 5. Block (2 von 10 = 20 %) ist Prüfung
GAP_FRAMES = 3            # Lücke zwischen Trainings- und Prüfblöcken

# ---------------------------------------------------------------- Gesichtsdetektor
try:
    import cv2
    _CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if _CASCADE.empty():
        _CASCADE = None
except Exception:                       # opencv fehlt oder Version 5 (dort ohne Haar-Kaskaden)
    _CASCADE = None
if CROP_MODE != "none" and _CASCADE is None:
    print("WARNUNG: Gesichtsdetektor fehlt -> ganzes Bild. Bitte: pip install \"opencv-python-headless<5\"")
    CROP_MODE = "none"
print(f"Bildausschnitt: {CROP_MODE}  (eyes = Augen, face = Gesicht, none = ganzes Bild)")


def detect_face(image: Image.Image):
    """Größtes Gesicht als (x, y, w, h) oder None."""
    gray = np.asarray(image.convert("L"))
    faces = _CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))


def crop_box(face, image_size, mode):
    """Fester Ausschnitt (x0, y0, x1, y1) aus der Gesichtsposition."""
    x, y, w, h = face
    W, H = image_size
    cx = x + w / 2
    if mode == "face":                    # Quadrat, 2,6 x Gesichtsbreite
        cy, bw, bh = y + h / 2, 2.6 * w, 2.6 * w
    else:                                 # "eyes": Streifen über beide Augen
        cy, bw, bh = y + 0.42 * h, 1.20 * w, 0.65 * w
    bw, bh = min(bw, W), min(bh, H)
    x0 = min(max(cx - bw / 2, 0), W - bw)
    y0 = min(max(cy - bh / 2, 0), H - bh)
    return (int(x0), int(y0), int(x0 + bw), int(y0 + bh))


def apply_crop(image, box, square=False):
    """Ausschnitt nehmen; mit square=True einen flachen Streifen mit Grau zum
    Quadrat ergänzen. Der Augenstreifen bleibt flach (STRIP_TF, 224 x 112)."""
    part = image.crop(box)
    if not square or abs(part.width - part.height) < 2:
        return part
    side = max(part.width, part.height)
    canvas = Image.new("RGB", (side, side), (124, 116, 104))   # ImageNet-Mittelwert
    canvas.paste(part, ((side - part.width) // 2, (side - part.height) // 2))
    return canvas


# ---------------------------------------------------------------- Modell
def fit_ridge(feats, Y, tr, va):
    """Ridge-Regression in dualer Form (geeignet für viele Merkmale, wenige Bilder).
    Gibt (Modell, Prüffehler, Faktor) zurück; die Stärke wird auf der Prüfung gewählt."""
    Fm, Yn = np.asarray(feats, np.float64), np.asarray(Y, np.float64)
    mu, sd = Fm[tr].mean(0), Fm[tr].std(0) + 1e-6
    Z = (Fm - mu) / sd
    ym = Yn[tr].mean(0)
    K = Z[tr] @ Z[tr].T
    scale = np.trace(K) / len(tr)
    best, results = (None, float("inf"), None), []
    for f in RIDGE_FACTORS:
        A = np.linalg.solve(K + f * scale * np.eye(len(tr)), Yn[tr] - ym)
        W = Z[tr].T @ A
        e = diagonal_error_pct(Yn[va], np.clip(Z[va] @ W + ym, 0, 1))
        results.append(((mu, sd, W, ym), e, f))
        if e < best[1]:
            best = ((mu, sd, W, ym), e, f)
    # Robuster wählen: die STÄRKSTE Regularisierung, die höchstens 5 % schlechter ist
    # als die beste. Sie extrapoliert weniger, wenn sich die Bilder bei der Verfolgung
    # etwas von der Kalibrierung unterscheiden (Licht, Kopfhaltung).
    ok = [r for r in results if r[1] <= best[1] * 1.05]
    return max(ok, key=lambda r: r[2])


def ridge_predict(model, feat):
    mu, sd, W, ym = model
    z = (np.asarray(feat, np.float64) - mu) / sd
    return np.clip(z @ W + ym, 0, 1).tolist()


def build_head():
    """Regressionskopf: 512 -> 512 -> 256 -> 128 -> 2, Sigmoid (identisch zu model.py)."""
    return nn.Sequential(
        nn.Linear(512, 512), nn.ReLU(),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 2), nn.Sigmoid(),
    )


def diagonal_error_pct(target, pred):
    """E_diag = 100 * sqrt(mean ||p - t||^2) / sqrt(2)  (wie metrics.py)."""
    t, p = np.asarray(target, float), np.asarray(pred, float)
    if len(t) == 0:
        return float("nan")
    return float(100 * np.sqrt(np.mean(np.sum((p - t) ** 2, axis=1))) / np.sqrt(2))


def block_split(n, n_blocks=N_BLOCKS, val_every=VAL_EVERY, gap=GAP_FRAMES):
    """Zeitliche Blöcke: jeder val_every-te Block = Prüfung (Block 2, 7, ...).
    Trainingsbilder näher als `gap` Bilder an einem Prüfbild werden weggelassen."""
    block = (np.arange(n) * n_blocks) // n
    is_val = (block % val_every) == (val_every // 2)
    va = np.flatnonzero(is_val)
    near = np.zeros(n, bool)
    for k in range(-gap, gap + 1):
        near |= np.roll(is_val, k) & (np.arange(n) + k >= 0) & (np.arange(n) + k < n)
    tr = np.flatnonzero(~is_val & ~near)
    return tr, va


class GazeModel:
    """Lädt das Basismodell einmal beim Start des Servers."""

    def __init__(self, model_file=MODEL_FILE):
        if not Path(model_file).exists():
            raise FileNotFoundError(f"Modelldatei fehlt: {model_file}")
        model = resnet18(weights=None)          # Gewichte kommen vollständig aus der Datei
        model.fc = build_head()
        model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
        model.eval()
        self.base_head = model.fc               # Kopf des Basismodells (bleibt unverändert)
        model.fc = nn.Identity()
        self.backbone = model                   # eingefrorener Merkmalsextraktor
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, image: Image.Image) -> torch.Tensor:
        """Ein Bild -> 512 Merkmale."""
        return self.backbone(EVAL_TF(image.convert("RGB")).unsqueeze(0))[0]

    @torch.no_grad()
    def predict(self, feat: torch.Tensor, head: nn.Module) -> list:
        head.eval()
        return head(feat.unsqueeze(0))[0].tolist()

    @torch.no_grad()
    def features_all(self, image: Image.Image, strip=False, pool=(4, 4)):
        """Ein Bild -> (512 globale Merkmale, räumliche layer3-Merkmale)."""
        m = self.backbone
        x = (STRIP_TF if strip else EVAL_TF)(image.convert("RGB")).unsqueeze(0)
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        l3 = m.layer3(m.layer2(m.layer1(x)))
        g = torch.flatten(m.avgpool(m.layer4(l3)), 1)[0]
        s = F.adaptive_avg_pool2d(l3, pool).flatten()
        return g, s

    @torch.no_grad()
    def base_predict(self, image: Image.Image) -> list:
        """Basismodell auf dem GANZEN Bild (wie beim Training auf GazeCapture)."""
        return self.predict(self.features(image), self.base_head)


class Session:
    """Daten und personalisierter Kopf EINER Nutzerin / eines Nutzers."""

    def __init__(self, gm: GazeModel, crop_mode=None):
        self.gm = gm
        self.crop_mode = crop_mode or CROP_MODE
        self.feats, self.targets, self.times, self.base_preds = [], [], [], []
        self.spatial = []                       # räumliche Merkmale (für Ridge)
        self.head = None                        # personalisierter Kopf (nach Kalibrierung)
        self.ridge = None                       # alternative lineare Abbildung
        self.method = "mlp"
        self.sridge = None
        self.error_after = None
        self.last_used = time.time()
        self.box = None                         # aktueller Ausschnitt dieser Sitzung
        self.face_size = None                   # feste Ausschnittgröße (erstes Gesicht)
        self.face_tries = 0
        self.n_pred = 0
        self.last_base = [0.5, 0.5]
        self.lock = threading.Lock()

    def prepare(self, image):
        """Ausschnitt anwenden. Die GRÖSSE wird beim ersten Gesicht festgelegt, die
        POSITION folgt dem Gesicht in jedem Bild. So bleiben die Augen im Ausschnitt,
        auch wenn sich der Kopf zwischen Kalibrierung und Verfolgung etwas bewegt."""
        if self.crop_mode == "none":
            return image
        face = detect_face(image) if (self.box is not None or self.face_tries < 30) else None
        with self.lock:
            if self.box is None:
                self.face_tries += 1
                if face is None:
                    return image
                self.face_size = (face[2], face[3])
            if face is not None:                          # neue Position, feste Größe
                fw, fh = self.face_size
                cx, cy = face[0] + face[2] / 2, face[1] + face[3] / 2
                self.box = crop_box((cx - fw / 2, cy - fh / 2, fw, fh), image.size, self.crop_mode)
            box = self.box
        return apply_crop(image, box, square=self.crop_mode == "face")

    def features(self, part):
        """(globale, räumliche) Merkmale des Ausschnitts."""
        return self.gm.features_all(part, strip=self.crop_mode == "eyes",
                                    pool=SPATIAL_POOL[self.crop_mode])

    def add_calibration_frame(self, image, x, y, t):
        """Kalibrierbild verarbeiten: Merkmale berechnen, das Bild selbst NICHT speichern."""
        self.last_used = time.time()
        if t < SKIP_START_SEC:
            return False
        part = self.prepare(image)
        if self.crop_mode != "none" and self.box is None:
            return False                                  # noch kein Gesicht gefunden
        with self.lock:
            idx = len(self.feats)
        feat, spat = self.features(part)
        base = None
        if self.crop_mode == "none":
            base = self.gm.predict(feat, self.gm.base_head)
        elif idx % BASE_EVERY == 0:
            base = self.gm.base_predict(image)
        if DEBUG_DIR is not None and idx % 30 == 0:
            DEBUG_DIR.mkdir(exist_ok=True)
            part.save(DEBUG_DIR / f"calib_{idx:03d}_t{t:05.1f}_x{x:.2f}_y{y:.2f}.jpg")
        with self.lock:
            self.feats.append(feat)
            self.spatial.append(spat)
            self.targets.append([float(x), float(y)])
            self.times.append(float(t))
            self.base_preds.append(base)
        return True

    def calibrate(self, seed=0, split=None):
        """Kopf auf den Kalibrierdaten trainieren. Gibt Kennzahlen zurück.

        split: optional (train_idx, val_idx) in zeitlicher Reihenfolge; nur für
        selftest.py (Vergleich mit dem zufälligen Split der Masterarbeit).
        """
        t0 = time.time()
        n = len(self.feats)
        if n < 40:
            raise ValueError(f"Zu wenige Kalibrierbilder ({n}). Wurde ein Gesicht erkannt? "
                             "Bitte gut ausgeleuchtet frontal in die Kamera schauen und neu kalibrieren.")
        order = np.argsort(self.times)                         # zeitliche Reihenfolge
        X = torch.stack([self.feats[i] for i in order])
        Y = torch.tensor([self.targets[i] for i in order], dtype=torch.float32)
        B = [self.base_preds[i] for i in order]
        rel = (X[1:] - X[:-1]).norm(dim=1) / X.norm(dim=1).mean().clamp_min(1e-9)
        if (rel < 1e-3).float().mean() > 0.5:
            raise ValueError("Die Kamerabilder haben sich während der Kalibrierung nicht verändert "
                             "(Video eingefroren). Bitte Seite neu laden und erneut kalibrieren.")
        tr, va = split if split is not None else block_split(n)

        # "Vorher": Basismodell auf ganzen Bildern, Prüfbilder mit berechneter Vorhersage
        vb = [i for i in va if B[i] is not None]
        err_before = diagonal_error_pct(Y[vb].numpy(), np.array([B[i] for i in vb])) if vb else None

        torch.manual_seed(seed)
        g = torch.Generator().manual_seed(seed)
        head = copy.deepcopy(self.gm.base_head)          # Start: Kopf des Basismodells
        for p in head.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.L1Loss()

        @torch.no_grad()
        def err(h, idx):
            h.eval()
            return diagonal_error_pct(Y[idx].numpy(), h(X[idx]).numpy())

        hist = {"train": [err(head, tr)], "val": [err(head, va)]}
        if err_before is None:
            err_before = hist["val"][0]
        best_err, best_ep, wait = hist["val"][0], 0, 0
        best_state = copy.deepcopy(head.state_dict())
        for ep in range(1, MAX_EPOCHS + 1):
            head.train()
            perm = torch.randperm(len(tr), generator=g)
            for i in range(0, len(tr), BATCH):
                idx = torch.as_tensor(tr)[perm[i:i + BATCH]]
                opt.zero_grad()
                loss_fn(head(X[idx]), Y[idx]).backward()
                opt.step()
            hist["train"].append(err(head, tr))
            hist["val"].append(err(head, va))
            if hist["val"][-1] < best_err:
                best_err, best_ep, wait = hist["val"][-1], ep, 0
                best_state = copy.deepcopy(head.state_dict())
            else:
                wait += 1
                if wait >= PATIENCE and ep >= MIN_EPOCHS:
                    break
        head.load_state_dict(best_state)
        head.eval()
        mlp_err = best_err

        # Ridge-Regression auf den globalen und auf den räumlichen Merkmalen
        S = torch.stack([self.spatial[i] for i in order]).numpy()
        ridge, ridge_err, ridge_alpha = fit_ridge(X.numpy(), Y.numpy(), tr, va)
        sridge, sridge_err, sridge_alpha = fit_ridge(S, Y.numpy(), tr, va)

        errors = {"mlp": mlp_err, "ridge": ridge_err, "spatial": sridge_err}
        self.method = min(errors, key=errors.get)            # bestes Verfahren auf der Prüfung
        self.head, self.ridge, self.sridge = head, ridge, sridge
        best_err = errors[self.method]
        self.error_after = best_err
        const = Y[tr].mean(0, keepdim=True).repeat(len(va), 1)  # Konstant-Baseline
        result = {
            "frames": n, "train_frames": int(len(tr)), "val_frames": int(len(va)),
            "error_before": round(err_before, 2), "error_after": round(best_err, 2),
            "error_constant": round(diagonal_error_pct(Y[va].numpy(), const.numpy()), 2),
            "best_epoch": best_ep, "epochs": len(hist["val"]) - 1, "seconds": round(time.time() - t0, 2),
            # Diagnose
            "train_before": round(hist["train"][0], 2), "train_last_epoch": round(hist["train"][-1], 2),
            "feature_change": round(float(rel.mean()), 4),
            "target_spread": round(float(Y.std(0).mean()), 3),
            "crop": self.crop_mode, "face_box": list(self.box) if self.box else None,
            "method": self.method, "error_mlp": round(mlp_err, 2),
            "error_ridge": round(ridge_err, 2), "ridge_alpha": ridge_alpha,
            "error_spatial": round(sridge_err, 2), "spatial_alpha": sridge_alpha,
            "spatial_dims": int(S.shape[1]),
        }
        if DEBUG_DIR is not None and split is None:
            DEBUG_DIR.mkdir(exist_ok=True)
            report = dict(result, history={k: [round(v, 2) for v in vals] for k, vals in hist.items()})
            (DEBUG_DIR / "calib_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        return result

    def predict(self, image):
        """Blickpunkt vorher (Basismodell, ganzes Bild) und nachher (personalisiert)."""
        self.last_used = time.time()
        part = self.prepare(image)
        feat, spat = self.features(part)
        if self.crop_mode == "none":
            base = self.gm.predict(feat, self.gm.base_head)
        else:
            if self.n_pred % BASE_EVERY == 0:
                self.last_base = self.gm.base_predict(image)
            base = self.last_base
        self.n_pred += 1
        return base, self.predict_personal(feat, spat)

    def predict_personal(self, feat, spat):
        if self.head is None:
            return None
        if self.method == "spatial":
            return ridge_predict(self.sridge, spat.numpy())
        if self.method == "ridge":
            return ridge_predict(self.ridge, feat.numpy())
        return self.gm.predict(feat, self.head)
