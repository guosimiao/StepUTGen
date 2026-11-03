import json
from pathlib import Path


def make_json_safe(obj):
    """
    Recursively convert an object into a JSON-serializable form.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, Exception):
        return str(obj)

    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)
    

def append_jsonl(record, jsonl_path):
    """
    Append one record to a JSONL file.
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")    