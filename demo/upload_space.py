"""Lädt den Ordner demo/ (ohne Modell) in einen Hugging Face Space hoch.

Hinweis: Docker-Spaces brauchen bei Hugging Face ein PRO-Abo. Das Modell wird nicht
mit hochgeladen; die Demo lädt es beim Start aus HF_MODEL_REPO (gaze_model.py).

Vorher einmal anmelden:   hf auth login      (ältere Versionen: huggingface-cli login)
Aufruf (im Ordner demo):  python upload_space.py
"""
from pathlib import Path

from huggingface_hub import HfApi

# >>> HIER ANPASSEN: <dein Hugging-Face-Name>/<Name des Space> <<<
REPO = "DEIN-NAME/gaze-personalization-demo"

HERE = Path(__file__).parent
if "DEIN-NAME" in REPO:
    raise SystemExit("Bitte zuerst REPO in upload_space.py anpassen.")

api = HfApi()
api.create_repo(REPO, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(HERE),
    repo_id=REPO,
    repo_type="space",
    ignore_patterns=[".venv/*", "venv/*", "__pycache__/*", "*.pyc", "selftest.py", "debug_frames/*",
                     "models/*", "upload_*.py", "MODEL_CARD.md"],   # Modell lädt die Demo selbst
    commit_message="Demo aktualisieren",
)
print("Fertig. Space:", f"https://huggingface.co/spaces/{REPO}")
print("Direkte Adresse:", f"https://{REPO.replace('/', '-').replace('_', '-').lower()}.hf.space")
