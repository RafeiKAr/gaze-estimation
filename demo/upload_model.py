"""Lädt NUR das Basismodell (models/base_norm_subject.pt) in ein Hugging-Face-Modell-Repository.

Vorher einmal anmelden:   hf auth login
Aufruf (im Ordner demo):  python upload_model.py RafeiKAr/<name-des-repositorys>

Danach den Namen des Repositorys in gaze_model.py bei HF_MODEL_REPO eintragen
(oder beim Start GAZE_MODEL_REPO setzen). Die Demo lädt das Modell dann selbst,
wenn es lokal fehlt.
"""
import sys
from pathlib import Path

from huggingface_hub import HfApi

HERE = Path(__file__).parent
MODEL = HERE / "models" / "base_norm_subject.pt"
CARD = HERE / "MODEL_CARD.md"

if len(sys.argv) != 2 or "/" not in sys.argv[1]:
    raise SystemExit("Aufruf: python upload_model.py <name>/<repository>")
if not MODEL.exists():
    raise SystemExit(f"Modelldatei fehlt: {MODEL}")
repo = sys.argv[1]

api = HfApi()
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=str(MODEL), path_in_repo=MODEL.name, repo_id=repo,
                repo_type="model", commit_message="Basismodell hochladen")
if CARD.exists():
    api.upload_file(path_or_fileobj=str(CARD), path_in_repo="README.md", repo_id=repo,
                    repo_type="model", commit_message="Modellbeschreibung")
print("Fertig:", f"https://huggingface.co/{repo}")
