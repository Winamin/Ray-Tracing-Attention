"""Local model selection and user data locations, independent of source paths."""
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
LOCAL=Path(os.environ.get('RT_HOME',str(ROOT/'.local'))).expanduser().resolve()

def default_model():
    if os.environ.get('RT_MODEL'): return Path(os.environ['RT_MODEL']).expanduser().resolve()
    selection=LOCAL/'active_model.json'
    if selection.exists(): return Path(json.loads(selection.read_text(encoding='utf8'))['directory'])
    return LOCAL/'models/default'

def select_model(directory):
    LOCAL.mkdir(parents=True,exist_ok=True)
    (LOCAL/'active_model.json').write_text(json.dumps({'directory':str(Path(directory).resolve())},indent=2),encoding='utf8')
