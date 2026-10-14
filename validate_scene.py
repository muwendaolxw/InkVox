#!/usr/bin/env python3
"""
Validate an InkVox scene JSON file against scene.schema.json.

Usage:
    python validate_scene.py scene.example.json
    python validate_scene.py path/to/my_scene.json
    python validate_scene.py my_scene.json --schema path/to/scene.schema.json

If --schema is not given, it looks for 'scene.schema.json' next to the
scene file, then next to this script.

Requires: pip install jsonschema
"""
import sys
import json
import argparse
from pathlib import Path

try:
    from jsonschema import Draft202012Validator
except ImportError:
    print("Missing dependency. Run:  pip install jsonschema")
    sys.exit(2)


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        # most common real-world failure: a trailing comma / typo in the JSON
        print(f"{path.name} is not valid JSON:")
        print(f"  line {e.lineno}, column {e.colno}: {e.msg}")
        sys.exit(1)


def find_schema(scene_path: Path, given: str | None) -> Path:
    if given:
        return Path(given)
    here = Path(__file__).resolve().parent
    data = load_json(scene_path)
    version = data.get("schema_version", "0.2")
    for candidate in (scene_path.parent / f"scene.schema.v{version}.json", here / f"scene.schema.v{version}.json", scene_path.parent / "scene.schema.json", here / "scene.schema.json"):
        if candidate.exists():
            return candidate
    print("Could not find scene.schema.json. Pass it with --schema.")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(description="Validate an InkVox scene against the schema.")
    ap.add_argument("scene", help="path to the scene JSON file to test")
    ap.add_argument("--schema", default=None, help="path to scene.schema.json")
    args = ap.parse_args()

    scene_path = Path(args.scene)
    schema_path = find_schema(scene_path, args.schema)

    schema = load_json(schema_path)
    scene = load_json(scene_path)

    # 1) make sure the schema itself is well-formed
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as e:
        print(f"The schema ({schema_path.name}) is itself invalid:")
        print(f"  {e}")
        sys.exit(2)

    # 2) validate the scene, collect ALL errors (not just the first)
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(scene), key=lambda e: list(e.path))

    from decision_state import consistency_errors
    semantic_errors=consistency_errors(scene.get('scene',{})) if scene.get('schema_version')=='0.4' else []
    if semantic_errors:
        for error in semantic_errors:print('FAIL  '+error)
    if not errors and not semantic_errors:
        print(f"PASS  {scene_path.name} conforms to {schema_path.name}")
        sys.exit(0)

    print(f"FAIL  {scene_path.name}: {len(errors)} problem(s)\n")
    for e in errors:
        where = " -> ".join(str(p) for p in e.path) or "(root)"
        print(f"  at: {where}")
        print(f"      {e.message}\n")
    sys.exit(1)


if __name__ == "__main__":
    main()
