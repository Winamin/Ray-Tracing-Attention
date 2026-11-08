"""Resolve the locally selected checkpoint for hardware verification."""
from ..runtime.settings import default_model
import json
from pathlib import Path

DEFAULT_OUTPUT=default_model()
def source_path():
    return Path(json.loads((DEFAULT_OUTPUT/'model.json').read_text(encoding='utf8'))['source_model'])
DEFAULT_GGUF=source_path() if (DEFAULT_OUTPUT/'model.json').exists() else None
