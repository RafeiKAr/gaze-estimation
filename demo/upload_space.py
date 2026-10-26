"""Lädt den Ordner demo/ in den Hugging Face Space hoch.

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
if not (HERE / "models" / "base_norm_subject.pt").exists():
    raise SystemExit("models/base_norm_subject.pt fehlt (siehe README.txt, Schritt 2).")

api = HfApi()
api.create_repo(REPO, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
api.upload_folder(
    folder_path=str(HERE),
    repo_id=REPO,
    repo_type="space",
    ignore_patterns=[".venv/*", "venv/*", "__pycache__/*", "*.pyc", "selftest.py", "README.txt", "ANLEITUNG_DEMO.md", "debug_frames/*"],
    commit_message="Demo aktualisieren",
)
print("Fertig. Space:", f"https://huggingface.co/spaces/{REPO}")
print("Direkte Adresse:", f"https://{REPO.replace('/', '-').replace('_', '-').lower()}.hf.space")
