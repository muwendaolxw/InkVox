#!/usr/bin/env python3
"""
structure_validator.py  --  InkVox structure validator (v0, fill-in stage)

Takes a raw scene JSON (as produced by the multimodal model, with fields left
empty) and fills in the gaps so it becomes a complete, renderable scene:

  * size       -- from a defaults knowledge base (by label, then by category);
                  a size_range is collapsed to its midpoint.
  * position   -- if missing and the object has a parent, it is placed on the
                  parent's top-center; otherwise it defaults to stage center.
  * rotation   -- defaults to [0, 0, 0].
  * confidence -- if missing, inferred from `source`
                  (voice/both high, sketch medium, inferred low).
  * source     -- defaults to "inferred".
  * coordinate_system / units -- ensured present.

Every value it supplies is recorded in scene.validation.filled so the user can
see exactly what was guessed.

NOTE: conflict detection (out-of-bounds / overlap) is the NEXT step and will be
added to this same module; for now status is set to "validated".

Usage:
    python structure_validator.py scene.raw.json
    python structure_validator.py scene.raw.json -o scene.filled.json
"""

import sys
import json
import math
import argparse
import re
from pathlib import Path
from copy import deepcopy

INKVOX_VALIDATOR_VERSION = "SCENE_V0_4_DESIGN_PARAMETERS_20261001"
MULTIVIEW_STRUCTURE_NORMALIZATION_V2 = True
INFO_FIDELITY_RELATION_LAYOUT_V3 = True

# ====================================================================
# DEFAULTS KNOWLEDGE BASE  --  extend this table as you add object types.
# Sizes are meters {w (X), d (Y), h (Z)}. Looked up by label first
# (case-insensitive), then by category.
# ====================================================================
DEFAULTS_BY_LABEL = {
    "throne":         {"w": 1.0, "d": 1.0, "h": 1.6},
    "chair":          {"w": 0.5, "d": 0.5, "h": 0.9},
    "table":          {"w": 1.2, "d": 0.8, "h": 0.75},
    "central riser":  {"w": 4.0, "d": 2.0, "h": 0.4},
    "led backdrop":   {"w": 10.0, "d": 0.2, "h": 5.0},
    "column":         {"w": 0.6, "d": 0.6, "h": 3.0},
    "pleated theatre curtain": {"w": 10.0, "d": 0.12, "h": 4.0},
    "overhead lighting truss": {"w": 8.0, "d": 0.35, "h": 0.35},
    "monitor wedge":  {"w": 0.5, "d": 0.35, "h": 0.25},
}
DEFAULTS_BY_CATEGORY = {
    "platform":          {"w": 3.0, "d": 2.0, "h": 0.3},
    "stairs":            {"w": 1.2, "d": 1.0, "h": 0.45},
    "prop":              {"w": 0.8, "d": 0.8, "h": 1.0},
    "furniture":         {"w": 0.8, "d": 0.8, "h": 1.0},
    "screen":            {"w": 6.0, "d": 0.2, "h": 3.5},
    "scenic_background": {"w": 10.0, "d": 0.12, "h": 4.0},
    "truss":             {"w": 8.0, "d": 0.35, "h": 0.35},
    "audio":             {"w": 0.6, "d": 0.4, "h": 0.3},
    "rigging":           {"w": 1.0, "d": 0.2, "h": 0.2},
    "curtain":           {"w": 10.0, "d": 0.12, "h": 4.0},
    "wall":              {"w": 10.0, "d": 0.2, "h": 4.2},
    "floor":             {"w": 12.0, "d": 8.0, "h": 0.03},
    "ramp":              {"w": 1.5, "d": 2.0, "h": 0.4},
    "set_piece":         {"w": 1.5, "d": 0.5, "h": 2.5},
    "decor":             {"w": 0.8, "d": 0.8, "h": 1.2},
    "rail":              {"w": 2.0, "d": 0.08, "h": 1.0},
    "column":            {"w": 0.5, "d": 0.5, "h": 3.0},
    "arch":              {"w": 3.0, "d": 0.4, "h": 3.0},
    "ceiling_piece":     {"w": 2.0, "d": 1.0, "h": 0.25},
    "floor_marking":     {"w": 1.0, "d": 1.0, "h": 0.01},
    "group_proxy":       {"w": 1.0, "d": 1.0, "h": 1.0},
    "unknown":           {"w": 1.0, "d": 1.0, "h": 1.0},
    # lights carry no bounding-box size by default
}

CONF_BY_SOURCE = {"voice": 0.9, "both": 0.9, "multi_view": 0.82, "canonical_render": 0.82, "front": 0.78, "front_left_45": 0.78, "front_right_45": 0.78, "semi_overhead": 0.82, "sketch": 0.55, "inferred": 0.3, "validator": 0.5, "user": 0.95, "unknown": 0.3}
DEFAULT_SOURCE = "inferred"

# v0.2 schema allows only these fields inside object.placement. Relation-like
# keys sometimes appear there in raw VLM output; the validator normalizes them.
PLACEMENT_ALLOWED_KEYS = {"anchor", "zone", "attached_to", "mounted_on", "side", "target", "faces", "coordinate_mode", "notes"}
PLACEMENT_RELATION_KEYS = {
    "behind": "behind",
    "in_front_of": "in_front_of",
    "left_of": "left_of",
    "right_of": "right_of",
    "on_top_of": "on_top_of",
    "centered_on": "centered_on",
    "aligned_with": "aligned_with",
    "light_focus": "light_focus",
    "aimed_at": "aimed_at",
    # v0.2 relation enum has no 'above'; use centered_on + overhead zone.
    "above": "centered_on",
}

# Scene schema v0.3 accepts only base object categories.  VLMs often emit
# semantic collection categories in scene.objects (e.g. light_array).  The
# validator legalizes those so filled JSON is schema-valid and keeps the array
# semantics in subtype / generation_hints for downstream builders.
CATEGORY_ALIASES = {
    "light_array": "light",
    "lighting_array": "light",
    "spotlight_array": "light",
    "lamp_array": "light",
    "audio_array": "audio",
    "speaker_array": "audio",
    "line_array": "audio",
    "furniture_array": "furniture",
    "chair_array": "furniture",
    "seating_array": "furniture",
    "stool_array": "furniture",
    "decor_array": "decor",
    "prop_array": "prop",
    "truss_array": "truss",
}

SHAPE_FAMILY_ALIASES = {
    "grid_truss": "box_truss",
    "truss_grid": "box_truss",
    "lighting_grid": "box_truss",
    "overhead_lighting_grid": "box_truss",
    "illuminated_glass_panel_wall": "scenic_wall",
    "glass_panel_wall": "scenic_wall",
    "backlit_glass_wall": "scenic_wall",
    "glowing_glass_wall": "scenic_wall",
    "perforated_metal_railing": "rail",
    "metal_railing": "rail",
    "seating_array": "unknown",
    "chair_array": "unknown",
}

ANCHOR_ALIASES = {
    "top_center": "center",
    "top": "center",
    "top_left": "custom",
    "top_right": "custom",
    "bottom_corner": "bottom_center",
    "bottom_left": "custom",
    "bottom_right": "custom",
    "front_left": "custom",
    "front_right": "custom",
    "rear_center": "back_center",
    "back": "back_center",
    "front": "front_center",
    "left": "left_edge",
    "right": "right_edge",
}

LIGHT_TYPE_ALIASES = {
    "ambient": "area",
    "environment": "area",
    "area_light": "area",
    "spotlight": "spot",
    "stage_spotlight": "spot",
}

# Conflict detection config.
PHYSICAL = {"platform", "prop", "screen", "stairs", "scenic_background", "curtain", "audio", "furniture", "wall", "floor", "ramp", "set_piece", "decor", "rail", "column", "arch", "ceiling_piece"}
OVERLAP_EPS = 1e-4   # interpenetration smaller than this counts as "just touching", not a conflict


# ====================================================================
# SCHEMA SANITIZER -- repairs common VLM raw-output slips so the filled
# scene can pass scene.schema.v0.3.json after validation.
# ====================================================================
SOURCE_KINDS = {"sketch", "voice", "canonical_render", "front", "front_left_45", "front_right_45", "semi_overhead", "multi_view", "inferred", "validator", "user", "unknown"}
REL_TYPES = {"on_top_of", "in_front_of", "behind", "left_of", "right_of", "above", "below", "centered_on", "aligned_with", "attached_to", "mounted_on", "aimed_at", "light_focus", "connected_to", "suspended_from", "supported_by", "surrounds", "nested_in", "part_of", "mirrored_with", "repeated_along", "parallel_to", "perpendicular_to", "faces", "distance", "no_occlusion", "occluded_by", "blocks_view_of"}
CONSTRAINT_TYPES = {"preserve_object", "preserve_count", "preserve_layout", "avoid_new_objects", "avoid_occlusion", "stage_safety", "user_intent", "visual_consistency", "unknown"}
PLACEMENT_SIDE_VALUES = {"left", "right", "center", "upstage", "downstage", "both", "unknown"}
VIS_VALUES = {"visible", "partially_visible", "occluded", "inferred", "not_visible", "unknown"}
CAMERA_ELEVATION = {"eye_level", "slightly_above", "high", "semi_overhead", "unknown"}
CAMERA_DISTANCE = {"close", "medium", "wide", "unknown"}


def _as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def _keep(d, keys):
    if not isinstance(d, dict):
        return {}
    return {k: v for k, v in d.items() if k in keys and v is not None}


def _norm_source(src, fallback="unknown"):
    if not src:
        return fallback
    s = str(src).strip()
    aliases = {
        "both": "multi_view",
        "image": "canonical_render",
        "render": "canonical_render",
        "model": "inferred",
        "default": "validator",
        "extractor_cleanup": "validator",
        "extractor_default": "validator",
        "cleanup": "validator",
        "vlm": "multi_view",
    }
    s = aliases.get(s, s)
    return s if s in SOURCE_KINDS else fallback



FOOTPRINT_TYPES = {"rectangle", "rounded_rectangle", "circle", "ellipse", "semicircle", "polygon", "bezier_polygon", "polyline", "composite_2d", "unknown"}
FOOTPRINT_SIDE_VALUES = {"downstage", "upstage", "left", "right", "front", "back", "unknown"}

def _norm_footprint_type(value, fallback="unknown"):
    """Normalize VLM footprint part labels into scene.schema.v0.3 footprint.type."""
    if value is None:
        return fallback
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "rect": "rectangle",
        "rectangular": "rectangle",
        "rectangular_back": "rectangle",
        "back_rectangle": "rectangle",
        "main_rectangle": "rectangle",
        "base_rectangle": "rectangle",
        "square": "rectangle",
        "round_rect": "rounded_rectangle",
        "rounded_rect": "rounded_rectangle",
        "circle_stage": "circle",
        "circular": "circle",
        "round": "circle",
        "oval": "ellipse",
        "semi_circle": "semicircle",
        "half_circle": "semicircle",
        "semicircular": "semicircle",
        "semicircular_apron": "semicircle",
        "downstage_apron": "semicircle",
        "apron": "semicircle",
        "arc": "semicircle",
        "poly": "polygon",
        "bezier": "bezier_polygon",
        "composite": "composite_2d",
        "rectangle_with_semicircular_apron": "composite_2d",
        "rectangular_stage_with_semicircular_apron": "composite_2d",
    }
    s = aliases.get(s, s)
    return s if s in FOOTPRINT_TYPES else fallback


def _num_or_none(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except Exception:
        return None


def _sanitize_vec2_list(points):
    out = []
    if not isinstance(points, list):
        return out
    for p in points:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            x, y = _num_or_none(p[0]), _num_or_none(p[1])
        elif isinstance(p, dict):
            x, y = _num_or_none(p.get("x")), _num_or_none(p.get("y"))
        else:
            continue
        if x is not None and y is not None:
            out.append([x, y])
    return out


def _sanitize_footprint(fp, size=None, default_type=None):
    """Recursively sanitize v0.3 footprint dictionaries.

    VLMs often output composite_2d.parts entries as {"shape":"rectangle"}
    or {"kind":"semicircular_apron"}.  The schema requires every footprint
    and every part to have a legal "type".  This normalizer is intentionally
    generic and applies equally to scene.stage.geometry.footprint and object
    geometry.footprint.
    """
    if not isinstance(fp, dict):
        fp = {}
    raw_type = fp.get("type") or fp.get("shape") or fp.get("kind") or fp.get("role") or default_type
    ftype = _norm_footprint_type(raw_type, "unknown")
    out = dict(fp)
    out["type"] = ftype
    # Avoid non-schema synonym confusion; additionalProperties are allowed, but
    # keeping a conflicting shape without type caused downstream confusion.
    if out.get("shape") == out.get("type"):
        out.pop("shape", None)

    # Common geometry aliases.
    if ftype == "ellipse":
        if out.get("radius_x") is None and out.get("rx") is not None:
            out["radius_x"] = out.get("rx")
        if out.get("radius_y") is None and out.get("ry") is not None:
            out["radius_y"] = out.get("ry")
        out.pop("rx", None); out.pop("ry", None)
    if ftype == "circle" and out.get("radius") is None:
        r = out.get("r") or out.get("radius_x") or out.get("radius_y")
        if r is None and isinstance(size, dict):
            w, d = _num_or_none(size.get("w")), _num_or_none(size.get("d"))
            if w and d:
                r = min(w, d) / 2.0
        if r is not None:
            out["radius"] = r
        out.pop("r", None)
    if ftype in {"rectangle", "rounded_rectangle"} and isinstance(size, dict):
        # Only infer missing width/depth; do not overwrite VLM dimensions.
        if out.get("w") is None and _num_or_none(size.get("w")) is not None:
            out["w"] = float(size.get("w"))
        if out.get("d") is None and _num_or_none(size.get("d")) is not None:
            out["d"] = float(size.get("d"))
    if ftype == "semicircle":
        side = str(out.get("side") or out.get("orientation") or "downstage").strip().lower()
        if side not in FOOTPRINT_SIDE_VALUES:
            if side in {"front", "audience", "down", "downstage_front"}:
                side = "downstage"
            elif side in {"rear", "back"}:
                side = "upstage"
            else:
                side = "downstage"
        out["side"] = side
        if out.get("radius") is None:
            r = out.get("r") or out.get("radius_x") or out.get("w")
            if _num_or_none(r) is not None and out.get("w") == r:
                # If only width is given for a semicircle, width is usually diameter.
                r = float(r) / 2.0
            if r is not None:
                out["radius"] = r
        out.pop("r", None)
    if ftype in {"polygon", "bezier_polygon", "polyline"}:
        pts = _sanitize_vec2_list(out.get("points"))
        if pts:
            out["points"] = pts
        if ftype != "polyline":
            out["closed"] = bool(out.get("closed", True))

    # Recursive composite support.
    raw_parts = out.get("parts")
    if ftype == "composite_2d" or isinstance(raw_parts, list):
        out["type"] = "composite_2d"
        parts = []
        for part in raw_parts or []:
            if not isinstance(part, dict):
                continue
            p_default = part.get("type") or part.get("shape") or part.get("kind") or part.get("role")
            # Infer common composite parts from labels.
            txt = json.dumps(part, ensure_ascii=False).lower()
            if not p_default:
                if any(k in txt for k in ("semi", "半圆", "apron", "arc", "curved")):
                    p_default = "semicircle"
                elif any(k in txt for k in ("rect", "矩形", "back", "rear")):
                    p_default = "rectangle"
            p = _sanitize_footprint(part, size=None, default_type=p_default or "unknown")
            parts.append(p)
        # If the VLM declared composite but forgot parts, synthesize a generic
        # rectangle + downstage semicircle from overall size.  This preserves the
        # intended topological shape without hard-coding any one scene.
        if not parts and isinstance(size, dict):
            w = _num_or_none(size.get("w")) or 6.0
            d = _num_or_none(size.get("d")) or 4.0
            apron_r = min(w / 2.0, d / 3.0)
            parts = [
                {"type": "rectangle", "w": w, "d": max(0.1, d - apron_r)},
                {"type": "semicircle", "side": "downstage", "radius": apron_r},
            ]
        out["parts"] = parts

    # Normalize source_views and confidence when present.
    if "source_views" in out:
        out["source_views"] = [_norm_source(x, "unknown") for x in _as_list(out.get("source_views"))]
    return out

def _sanitize_source_evidence_list(items):
    out = []
    for e in _as_list(items):
        if not isinstance(e, dict):
            continue
        src = _norm_source(e.get("source"), "unknown")
        supports = e.get("supports")
        if isinstance(supports, str):
            supports = [supports]
        if not isinstance(supports, list):
            supports = []
        ne = {"source": src, "supports": [str(x) for x in supports], "confidence": e.get("confidence", 0.5)}
        if e.get("notes") is not None:
            ne["notes"] = str(e.get("notes"))
        out.append(ne)
    return out


def _sanitize_coordinate_system(cs):
    if not isinstance(cs, dict):
        cs = {}
    return {
        "units": "m",
        "angle_units": "deg",
        "origin": "downstage_center_floor",
        "x": "audience_right",
        "y": "upstage_depth",
        "z": "up",
    }


def _sanitize_stage(stage):
    if not isinstance(stage, dict):
        stage = {}
    out = _keep(stage, {"width", "depth", "height_clearance", "shape", "footprint", "material", "geometry", "source_evidence"})
    out.setdefault("width", 12)
    out.setdefault("depth", 8)
    out.setdefault("height_clearance", 6)
    _normalize_material_color_fields(out, stage)
    stage_size = {"w": out.get("width"), "d": out.get("depth"), "h": out.get("height") or 0.3}
    if isinstance(out.get("footprint"), dict):
        out["footprint"] = _sanitize_footprint(out["footprint"], size=stage_size, default_type=out.get("shape"))
    if isinstance(out.get("geometry"), dict):
        out["geometry"] = _sanitize_geometry(out["geometry"], "platform", out.get("shape"), stage_size)
        if isinstance(out.get("geometry"), dict) and isinstance(out["geometry"].get("footprint"), dict):
            out["footprint"] = _sanitize_footprint(out["geometry"].get("footprint"), size=stage_size, default_type=out.get("shape"))
    if "source_evidence" in out:
        out["source_evidence"] = _sanitize_source_evidence_list(out["source_evidence"])
    return out




def _infer_stage_subtype(raw_stage):
    text = " ".join(str(raw_stage.get(k, "")) for k in ("id", "label", "subtype", "description", "shape")).lower()
    geom = raw_stage.get("geometry") if isinstance(raw_stage.get("geometry"), dict) else {}
    fp = geom.get("footprint") if isinstance(geom.get("footprint"), dict) else raw_stage.get("footprint")
    fp_type = fp.get("type") if isinstance(fp, dict) else None
    fp_blob = json.dumps(fp, ensure_ascii=False).lower() if isinstance(fp, dict) else ""
    if fp_type == "composite_2d" or "semicircular_apron" in fp_blob or "rectangle_with_semicircular" in text:
        return "rectangular_stage_with_semicircular_apron", "composite_2d"
    if any(x in text for x in ("circle", "circular", "round", "圆", "圆形")) or fp_type == "circle":
        return "circular_stage", "circle"
    if any(x in text for x in ("oval", "ellipse", "椭圆")) or fp_type in ("oval", "ellipse"):
        return "oval_stage", "ellipse"
    if any(x in text for x in ("semicircle", "semi_circle", "半圆")) or fp_type == "semicircle":
        return "semicircular_stage", "semicircle"
    return raw_stage.get("subtype") or "main_stage", raw_stage.get("shape") or geom.get("shape") or "rectangle"


def _stage_size_from_raw(raw_stage):
    size = raw_stage.get("size") if isinstance(raw_stage.get("size"), dict) else {}
    w = size.get("w") or raw_stage.get("w") or raw_stage.get("width")
    d = size.get("d") or raw_stage.get("d") or raw_stage.get("depth")
    h = size.get("h") or raw_stage.get("h") or raw_stage.get("height") or raw_stage.get("platform_height") or 0.3
    def num(v, default):
        try:
            return float(v)
        except Exception:
            return default
    return {"w": num(w, 6.0), "d": num(d, 4.0), "h": num(h, 0.3)}


def _stage_object_candidate(raw_stage, existing_objects=None, raw_relations=None):
    """Recover a platform object when a VLM incorrectly placed the physical
    main stage inside scene.stage instead of scene.objects. This is conservative:
    it only fires when stage contains object-like evidence or relations/labels
    clearly reference a main stage object while no platform object exists.
    """
    if not isinstance(raw_stage, dict):
        return None
    existing_objects = existing_objects or []
    if any(isinstance(o, dict) and o.get("category") == "platform" for o in existing_objects):
        return None
    blob = json.dumps({"stage": raw_stage, "relations": raw_relations or []}, ensure_ascii=False).lower()
    object_like = (
        raw_stage.get("category") == "platform" or
        any(k in raw_stage for k in ("id", "label", "subtype", "description", "placement", "visibility")) or
        any(x in blob for x in ("main_stage", "circular_stage", "wooden stage", "圆形舞台", "主舞台", "平台"))
    )
    if not object_like:
        return None
    subtype, shape = _infer_stage_subtype(raw_stage)
    oid = raw_stage.get("id") or "main_stage"
    oid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(oid)).strip("_").lower() or "main_stage"
    if oid in {o.get("id") for o in existing_objects if isinstance(o, dict)}:
        oid = "main_stage"
    size = _stage_size_from_raw(raw_stage)
    geom = raw_stage.get("geometry") if isinstance(raw_stage.get("geometry"), dict) else {}
    if not geom:
        footprint = None
        if shape == "circle":
            footprint = {"type": "circle", "radius": round(min(size["w"], size["d"]) / 2.0, 3)}
        elif shape in ("oval", "ellipse"):
            footprint = {"type": "ellipse", "rx": round(size["w"] / 2.0, 3), "ry": round(size["d"] / 2.0, 3)}
        else:
            footprint = {"type": "rectangle", "w": size["w"], "d": size["d"]}
        geom = {
            "representation": "contour_extrusion",
            "shape": shape,
            "footprint": footprint,
            "extrude_height": size["h"],
            "coordinate_mode": "bottom_center",
        }
    material = raw_stage.get("material") if isinstance(raw_stage.get("material"), dict) else None
    if not material:
        text = json.dumps(raw_stage, ensure_ascii=False).lower()
        material = {"type": "wood", "color_name": "warm_wood"} if ("wood" in text or "木" in text) else {"type": "painted_stage", "color_name": "warm_neutral"}
    placement = raw_stage.get("placement") if isinstance(raw_stage.get("placement"), dict) else {"zone": "center", "anchor": "bottom_center"}
    position = raw_stage.get("position") if isinstance(raw_stage.get("position"), list) and len(raw_stage.get("position")) == 3 else [0.0, round(size["d"] / 2.0, 3), 0.0]
    obj = {
        "id": oid,
        "category": "platform",
        "subtype": subtype,
        "label": raw_stage.get("label") or raw_stage.get("name") or "main stage platform",
        "position": position,
        "size": size,
        "geometry": geom,
        "material": material,
        "placement": placement,
        "source": _norm_source(raw_stage.get("source"), "inferred"),
        "confidence": raw_stage.get("confidence", 0.55),
        "source_views": raw_stage.get("source_views", ["inferred"]),
        "source_evidence": _sanitize_source_evidence_list(raw_stage.get("source_evidence")),
    }
    return _sanitize_object(obj)

def _sanitize_visibility(vis):
    if not isinstance(vis, dict):
        return vis if vis is None else {"overall": "unknown"}
    allowed = {"overall", "by_view", "inferred_parts"}
    if any(k in vis for k in allowed):
        out = {}
        overall = vis.get("overall") or vis.get("status")
        if overall in VIS_VALUES:
            out["overall"] = overall
        by = vis.get("by_view") if isinstance(vis.get("by_view"), dict) else {}
        for k, v in vis.items():
            if k in SOURCE_KINDS and v in VIS_VALUES:
                by[k] = v
        if by:
            out["by_view"] = {str(k): (v if v in VIS_VALUES else "unknown") for k, v in by.items()}
        parts = vis.get("inferred_parts")
        if isinstance(parts, list):
            out["inferred_parts"] = [str(x) for x in parts]
        return out or {"overall": "unknown"}
    by = {}
    overall = vis.get("status") if vis.get("status") in VIS_VALUES else None
    for k, v in vis.items():
        if k in SOURCE_KINDS and v in VIS_VALUES:
            by[k] = v
    out = {}
    if overall:
        out["overall"] = overall
    elif by:
        vals = list(by.values())
        out["overall"] = "visible" if "visible" in vals else vals[0]
    if by:
        out["by_view"] = by
    return out or {"overall": "unknown"}




def _norm_shape_family(value, fallback="unknown"):
    if not value:
        return fallback
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    s = SHAPE_FAMILY_ALIASES.get(s, s)
    allowed = {"straight_stairs", "curved_stairs", "tiered_platform", "circular_stage", "semi_circular_apron", "pleated_curtain", "box_truss", "ladder_truss", "stage_spotlight", "wash_light", "monitor_wedge", "led_screen_wall", "scenic_wall", "arch", "column_pair", "rail", "unknown"}
    return s if s in allowed else fallback


def _sanitize_light_attributes(attrs):
    if not isinstance(attrs, dict):
        return attrs
    out = deepcopy(attrs)
    light = out.get("light")
    if isinstance(light, dict):
        ltype = light.get("type")
        if ltype is not None:
            key = str(ltype).strip().lower().replace("-", "_").replace(" ", "_")
            light["type"] = LIGHT_TYPE_ALIASES.get(key, key)
            if light["type"] not in {"spot", "wash", "par", "beam", "flood", "point", "area", "unknown"}:
                light["type"] = "unknown"
        out["light"] = light
    return out


def _sanitize_geometry(geom, category=None, subtype=None, size=None):
    """Normalize v0.2-style geometry into v0.3 geometry schema."""
    if not isinstance(geom, dict):
        return geom
    shape = str(geom.get("shape") or geom.get("type") or "").strip()
    # Normalize procedural shape families before deciding representation.
    raw_proc = geom.get("procedural") if isinstance(geom.get("procedural"), dict) else {}
    proc_family = _norm_shape_family(raw_proc.get("shape_family") or geom.get("shape_family"), None)
    if proc_family and proc_family != "unknown" and (not shape or shape == "custom"):
        shape = proc_family
    low = (shape + " " + str(category or "") + " " + str(subtype or "") + " " + str(proc_family or "")).lower()
    out = {"shape": shape or "unknown"}
    coord = geom.get("coordinate_mode") or geom.get("anchor")
    if coord in {"bottom_center", "center", "custom_anchor", "image_plane", "unknown"}:
        out["coordinate_mode"] = coord
    ff = geom.get("front_face")
    if ff in {"+x", "-x", "+y", "-y", "+z", "-z", "audience", "unknown"}:
        out["front_face"] = ff
    if geom.get("confidence") is not None:
        out["confidence"] = geom.get("confidence")

    procedural_map = {
        "stairs": "straight_stairs",
        "straight_stairs": "straight_stairs",
        "pleated_surface": "pleated_curtain",
        "pleated_curtain": "pleated_curtain",
        "truss_box": "box_truss",
        "box_truss": "box_truss",
        "grid_truss": "box_truss",
        "truss_grid": "box_truss",
        "lighting_grid": "box_truss",
        "overhead_lighting_grid": "box_truss",
        "stage_spotlight": "stage_spotlight",
        "spotlight": "stage_spotlight",
        "wash_light": "wash_light",
        "monitor_wedge": "monitor_wedge",
        "line_array": "unknown",
        "led_screen_wall": "led_screen_wall",
        "scenic_wall": "scenic_wall",
        "wall_panel": "scenic_wall",
        "illuminated_glass_panel_wall": "scenic_wall",
        "glass_panel_wall": "scenic_wall",
        "backlit_glass_wall": "scenic_wall",
        "perforated_metal_railing": "rail",
        "metal_railing": "rail",
    }
    # Let category/subtype rescue missing/old shape labels.
    if category == "stairs" and shape in ("", "stairs"):
        low = "straight_stairs " + low
    if category == "curtain" or "curtain" in low:
        if shape in ("", "custom", "pleated_surface"):
            low = "pleated_surface " + low
    if category == "truss" and shape in ("", "custom"):
        low = "truss_box " + low
    if category == "light" and shape in ("", "custom"):
        low = "stage_spotlight " + low
    if category == "audio" and ("monitor" in low or "wedge" in low) and shape in ("", "custom"):
        low = "monitor_wedge " + low

    family = None
    for key, fam in procedural_map.items():
        if key in low:
            family = fam
            break
    if family:
        out["representation"] = "procedural"
        out["procedural"] = {"shape_family": _norm_shape_family(family, family)}
        for k in ("step_count", "step_rise", "step_run", "pleat_count", "pleat_depth", "segment_count", "cross_section", "diagonal_pattern", "top_valance", "side_legs", "detail_params"):
            if geom.get(k) is not None:
                out["procedural"][k] = geom.get(k)
            elif isinstance(raw_proc, dict) and raw_proc.get(k) is not None:
                out["procedural"][k] = raw_proc.get(k)
        return out

    if geom.get("representation") in {"primitive", "composite", "contour_extrusion", "procedural", "asset_ref", "proxy", "unknown"}:
        # Keep only schema-allowed fields from already-v0.3 geometry, while
        # recursively legalizing footprint/composite_2d parts.
        allowed = {"representation", "shape", "primitive", "footprint", "extrude_height", "composite_parts", "procedural", "front_face", "coordinate_mode", "detail_params", "confidence", "source_views"}
        kept = _keep(geom, allowed)
        if isinstance(kept.get("procedural"), dict):
            kept["procedural"] = dict(kept["procedural"])
            kept["procedural"]["shape_family"] = _norm_shape_family(kept["procedural"].get("shape_family"), "unknown")
        if isinstance(kept.get("footprint"), dict):
            kept["footprint"] = _sanitize_footprint(kept["footprint"], size=size, default_type=kept.get("shape"))
            if kept["footprint"].get("type") == "composite_2d":
                kept["shape"] = "composite_2d"
                kept["representation"] = "contour_extrusion"
        if "source_views" in kept:
            kept["source_views"] = [_norm_source(x, "unknown") for x in _as_list(kept["source_views"])]
        return kept

    primitive_shapes = {"box", "cylinder", "sphere", "plane", "cone", "torus", "curve", "mesh_proxy"}
    if shape in primitive_shapes:
        out["representation"] = "primitive"
        out["primitive"] = {"type": shape}
        if geom.get("segments") is not None:
            out["primitive"]["segments"] = geom.get("segments")
        if geom.get("bevel") is not None:
            out["primitive"]["bevel"] = geom.get("bevel")
        return out

    contour_shapes = {"circle", "oval", "ellipse", "rectangle", "rounded_rectangle", "semicircle", "polygon", "bezier_polygon", "polyline", "composite_2d", "arc"}
    if shape in contour_shapes or category == "platform":
        fp = geom.get("footprint") if isinstance(geom.get("footprint"), dict) else {}
        if not fp:
            if shape == "circle":
                r = None
                if isinstance(size, dict):
                    try: r = min(float(size.get("w", 0)), float(size.get("d", 0))) / 2.0
                    except Exception: r = None
                fp = {"type": "circle", "radius": round(r or float(geom.get("radius", 1.5) or 1.5), 3)}
            elif shape in ("oval", "ellipse"):
                fp = {"type": "ellipse"}
                if isinstance(size, dict):
                    try:
                        fp["radius_x"] = round(float(size.get("w", 2.0)) / 2.0, 3)
                        fp["radius_y"] = round(float(size.get("d", 1.5)) / 2.0, 3)
                    except Exception:
                        pass
            elif shape == "semicircle" or shape == "arc":
                fp = {"type": "semicircle", "side": geom.get("side", "downstage") if geom.get("side") in {"downstage", "upstage", "left", "right", "front", "back", "unknown"} else "downstage"}
            elif shape == "polygon" and isinstance(geom.get("points"), list):
                fp = {"type": "polygon", "points": geom.get("points"), "closed": bool(geom.get("closed", True))}
            else:
                fp = {"type": "rectangle"}
                if isinstance(size, dict):
                    try:
                        fp["w"] = float(size.get("w", 1.0)); fp["d"] = float(size.get("d", 1.0))
                    except Exception:
                        pass
        fp = _sanitize_footprint(fp, size=size, default_type=shape or ("rectangle" if category == "platform" else None))
        if fp.get("type") == "composite_2d":
            out["shape"] = "composite_2d"
        out["representation"] = "contour_extrusion"
        out["footprint"] = fp
        if isinstance(size, dict) and size.get("h"):
            out["extrude_height"] = size.get("h")
        elif geom.get("extrude_height"):
            out["extrude_height"] = geom.get("extrude_height")
        return out

    out["representation"] = "unknown"
    return out

def _sanitize_placement(pl):
    if not isinstance(pl, dict):
        return pl
    out = _keep(pl, {"anchor", "zone", "side", "attached_to", "mounted_on", "target", "faces", "coordinate_mode", "notes"})
    anchor = out.get("anchor")
    if anchor is not None:
        key = str(anchor).strip().lower().replace("-", "_").replace(" ", "_")
        out["anchor"] = ANCHOR_ALIASES.get(key, key)
        if out["anchor"] not in {"bottom_center", "center", "front_center", "back_center", "left_edge", "right_edge", "custom", "unknown"}:
            out["anchor"] = "unknown"
    side = out.get("side")
    if side not in (None, *PLACEMENT_SIDE_VALUES):
        low = str(side).lower()
        if low in ("front", "down", "downstage_front"):
            out["side"] = "downstage"
        elif low in ("back", "rear", "upstage_back"):
            out["side"] = "upstage"
        elif "left" in low:
            out["side"] = "left"
        elif "right" in low:
            out["side"] = "right"
        elif "center" in low or "middle" in low or "above" in low:
            out["side"] = "center"
        else:
            out["side"] = "unknown"
    return out


def _normalize_material_color_fields(out, raw=None):
    """Preserve VLM material/color evidence in a schema-friendly form.

    MULTIVIEW_STRUCTURE_NORMALIZATION_V2: raw JSON often has
    material: "fabric" and color: "purple".  Earlier sanitizer kept only the
    material string and lost the color, which made Blender outputs grey/white.
    """
    raw = raw or out
    color = out.pop("color", None)
    if color is None and isinstance(raw, dict):
        color = raw.get("color_name") or raw.get("base_color")
    mat = out.get("material")
    if isinstance(mat, dict):
        mat = dict(mat)
        legacy_color = mat.pop("color", None)
        if legacy_color is not None:
            if isinstance(legacy_color, str) and re.fullmatch(r'#[0-9a-fA-F]{6}', legacy_color):
                if not mat.get("color_hex"):mat["color_hex"] = legacy_color
            elif not mat.get("color_name"):
                mat["color_name"] = str(legacy_color)
        for key in ("type", "color_hex", "color_name"):
            if mat.get(key) is None:mat.pop(key, None)
        if color is not None and not any(mat.get(k) for k in ("color_name", "color_hex", "base_color", "color")):
            mat["color_name"] = str(color)
        out["material"] = mat
    elif isinstance(mat, str):
        mat_obj = {"type": mat}
        if color is not None:
            mat_obj["color_name"] = str(color)
        out["material"] = mat_obj
    elif color is not None:
        out["material"] = {"type": "unknown", "color_name": str(color)}
    return out


def _sanitize_object(obj):
    if not isinstance(obj, dict):
        return None
    allowed = {"object_notes", "parameter_info", "parameter_states", "id", "category", "subtype", "label", "description", "parent", "group_id", "position", "rotation", "size", "size_range", "relative_size", "geometry", "material", "color", "placement", "visibility", "attributes", "components", "asset", "generation_hints", "confidence", "source", "source_views", "source_evidence"}
    out = _keep(obj, allowed)
    out.setdefault("label", str(out.get("id", "Design object")).replace("_", " "))
    # Preserve compact array/layout/prototype information without relying on
    # non-schema top-level fields. generate_stage.py reads count from
    # generation_hints, and later validator passes can use the original layout.
    gh0 = out.get("generation_hints") if isinstance(out.get("generation_hints"), dict) else {}
    if obj.get("count") is not None and gh0.get("count") is None:
        gh0["count"] = obj.get("count")
    if isinstance(obj.get("layout"), dict) and gh0.get("layout") is None:
        gh0["layout"] = deepcopy(obj.get("layout"))
    if isinstance(obj.get("prototype"), dict) and gh0.get("prototype") is None:
        gh0["prototype"] = _sanitize_prototype(obj.get("prototype"))
    if gh0:
        out["generation_hints"] = gh0
    _normalize_material_color_fields(out, obj)
    raw_cat = out.get("category")
    if isinstance(raw_cat, str):
        key = raw_cat.strip().lower().replace("-", "_").replace(" ", "_")
        legal_cat = CATEGORY_ALIASES.get(key, key)
        if legal_cat != raw_cat:
            hints = out.get("generation_hints") if isinstance(out.get("generation_hints"), dict) else {}
            hints["raw_category"] = raw_cat
            if "array" in key or "group" in key:
                hints["treat_as_group"] = True
            out["generation_hints"] = hints
            if not out.get("subtype"):
                out["subtype"] = key
            out["category"] = legal_cat
    if "source" in out:
        out["source"] = _norm_source(out.get("source"), "unknown")
    if "attributes" in out:
        out["attributes"] = _sanitize_light_attributes(out["attributes"])
    if "geometry" in out:
        out["geometry"] = _sanitize_geometry(out["geometry"], out.get("category"), out.get("subtype"), out.get("size"))
    if out.get("category") == "platform" and isinstance(out.get("geometry"), dict):
        fp = out["geometry"].get("footprint") if isinstance(out["geometry"].get("footprint"), dict) else {}
        fp_blob = json.dumps(fp, ensure_ascii=False).lower()
        if fp.get("type") == "composite_2d" and ("semicircle" in fp_blob or "apron" in fp_blob):
            if out.get("subtype") in (None, "circular_stage", "main_stage", "platform") or "circular" in str(out.get("subtype", "")).lower():
                out["subtype"] = "rectangular_stage_with_semicircular_apron"
            if out.get("label") and "circular" in str(out.get("label", "")).lower():
                out["label"] = "main stage platform with semicircular apron"
    if "placement" in out:
        out["placement"] = _sanitize_placement(out["placement"])
    if "visibility" in out:
        out["visibility"] = _sanitize_visibility(out["visibility"])
    if "source_evidence" in out:
        out["source_evidence"] = _sanitize_source_evidence_list(out["source_evidence"])
    return out


def _sanitize_prototype(proto):
    if not isinstance(proto, dict):
        return {}
    allowed = {"category", "subtype", "label", "size", "geometry", "material", "color", "placement", "attributes", "asset", "generation_hints"}
    out = _keep(proto, allowed)
    _normalize_material_color_fields(out, proto)
    raw_cat = out.get("category")
    if isinstance(raw_cat, str):
        key = raw_cat.strip().lower().replace("-", "_").replace(" ", "_")
        out["category"] = CATEGORY_ALIASES.get(key, key)
        if not out.get("subtype") and out["category"] != key:
            out["subtype"] = key
    if "attributes" in out:
        out["attributes"] = _sanitize_light_attributes(out["attributes"])
    if "geometry" in out:
        out["geometry"] = _sanitize_geometry(out["geometry"], out.get("category"), out.get("subtype"), out.get("size"))
    if "placement" in out:
        out["placement"] = _sanitize_placement(out["placement"])
    return out


def _sanitize_group(group):
    if not isinstance(group, dict):
        return None
    allowed = {"id", "category", "label", "count", "prototype", "distribution", "layout", "expanded_object_ids", "confidence", "source_evidence"}
    out = _keep(group, allowed)
    if isinstance(out.get("layout"), dict) and not isinstance(out.get("distribution"), dict):
        out["distribution"] = deepcopy(out.get("layout"))
    out.pop("layout", None)
    if isinstance(out.get("category"), str):
        key = out["category"].strip().lower().replace("-", "_").replace(" ", "_")
        out["category"] = CATEGORY_ALIASES.get(key, key)
    out["prototype"] = _sanitize_prototype(out.get("prototype") or {})
    if "source_evidence" in out:
        out["source_evidence"] = _sanitize_source_evidence_list(out["source_evidence"])
    return out


def _parse_relation_text(text):
    if not isinstance(text, str):
        return None, None
    t = text.strip().replace(":", " ")
    low = t.lower()
    aliases = {
        "in front of": "in_front_of", "in_front_of": "in_front_of", "behind": "behind",
        "left of": "left_of", "left_of": "left_of", "right of": "right_of", "right_of": "right_of",
        "attached to": "attached_to", "attached_to": "attached_to", "mounted on": "mounted_on", "mounted_on": "mounted_on",
        "aimed at": "aimed_at", "aimed_at": "aimed_at", "light focus": "light_focus", "light_focus": "light_focus",
        "on top of": "on_top_of", "on_top_of": "on_top_of", "above": "above", "centered on": "centered_on", "centered_on": "centered_on",
    }
    for phrase, rtype in sorted(aliases.items(), key=lambda x: -len(x[0])):
        if low.startswith(phrase):
            target = t[len(phrase):].strip(" _:-") or None
            return rtype, target
    parts = t.split()
    if parts and parts[0] in REL_TYPES:
        return parts[0], (parts[1] if len(parts) > 1 else None)
    return None, None


def _sanitize_relation(rel):
    if not isinstance(rel, dict):
        return None
    # MULTIVIEW_STRUCTURE_NORMALIZATION_V2:
    # VLMs often use object/target/to/attached_to/mounted_on as the relation
    # target.  Do not drop these aliases; downstream layout depends on them.
    rtype = rel.get("type") or rel.get("relation_type")
    subject = rel.get("subject") or rel.get("from") or rel.get("source_object")
    target = (rel.get("target") or rel.get("object") or rel.get("to") or
              rel.get("target_id") or rel.get("attached_to") or rel.get("mounted_on"))
    if isinstance(rel.get("relation"), str):
        parsed_type, parsed_target = _parse_relation_text(rel.get("relation"))
        rtype = rtype or parsed_type
        target = target or parsed_target
    if rtype not in REL_TYPES:
        # Keep weak relation evidence rather than producing schema-invalid output.
        rtype = "connected_to" if subject else None
    if not rtype or not subject:
        return None
    out = {"type": str(rtype), "subject": str(subject)}
    if target is not None:
        out["target"] = str(target)
    if isinstance(rel.get("params"), dict):
        out["params"] = rel["params"]
    if rel.get("confidence") is not None:
        out["confidence"] = rel.get("confidence")
    out["source"] = _norm_source(rel.get("source"), "unknown")
    if "source_evidence" in rel:
        out["source_evidence"] = _sanitize_source_evidence_list(rel["source_evidence"])
    return out


def _sanitize_constraint(c):
    if not isinstance(c, dict):
        return None
    ctype = c.get("type")
    if ctype not in CONSTRAINT_TYPES:
        # relation-specific requirements are still user intent / visual consistency.
        if str(ctype).startswith("require_") or str(ctype).startswith("preserve_"):
            ctype = "user_intent"
        else:
            ctype = "unknown"
    msg = c.get("message")
    if not msg:
        field = c.get("field")
        value = c.get("value")
        if field is not None and value is not None:
            msg = f"Preserve {field} = {value}."
        elif field is not None:
            msg = f"Preserve or satisfy constraint on {field}."
        else:
            msg = f"Constraint generated by extractor: {ctype}."
    out = {"type": ctype, "message": str(msg)}
    objs = c.get("objects")
    if isinstance(objs, str):
        out["objects"] = [objs]
    elif isinstance(objs, list):
        out["objects"] = [str(x) for x in objs]
    if c.get("priority") in {"low", "medium", "high", "critical"}:
        out["priority"] = c.get("priority")
    out["source"] = _norm_source(c.get("source"), "unknown")
    if c.get("confidence") is not None:
        out["confidence"] = c.get("confidence")
    return out


def _sanitize_camera(cam, idx=0):
    if not isinstance(cam, dict):
        return None
    out = _keep(cam, {"id", "view_name", "position", "target", "elevation", "distance", "focal_length", "source_image", "notes"})
    out.setdefault("id", f"camera_{idx+1}")
    out["view_name"] = _norm_source(out.get("view_name") or cam.get("source"), "canonical_render")
    if out.get("elevation") not in (None, *CAMERA_ELEVATION):
        out["elevation"] = "unknown"
    if out.get("distance") not in (None, *CAMERA_DISTANCE):
        out["distance"] = "unknown"
    if "focal_length" not in out and cam.get("field_of_view_deg") is not None:
        out["notes"] = (out.get("notes", "") + f" field_of_view_deg={cam.get('field_of_view_deg')}").strip()
    return out


def _sanitize_validation(val):
    if not isinstance(val, dict):
        return {"status": "raw", "conflicts": [], "filled": [], "unresolved_fields": []}
    out = _keep(val, {"status", "conflicts", "filled", "unresolved_fields"})
    if out.get("status") not in {"raw", "validated", "has_conflicts", "partial"}:
        out["status"] = "raw"
    out.setdefault("conflicts", [])
    out.setdefault("filled", [])
    out.setdefault("unresolved_fields", [])
    return out



def _sync_stage_footprint_from_main_stage(scene):
    """Keep scene.stage footprint/geometry consistent with main platform.

    Multi-view extraction may correctly put the detailed composite footprint on
    main_stage while leaving scene.stage.geometry with the same raw invalid
    parts.  This keeps both schema-valid and semantically aligned.
    """
    if not isinstance(scene, dict):
        return
    objects = scene.get("objects") if isinstance(scene.get("objects"), list) else []
    main = None
    for o in objects:
        if not isinstance(o, dict):
            continue
        if o.get("id") == "main_stage" or (main is None and o.get("category") == "platform"):
            main = o
            if o.get("id") == "main_stage":
                break
    if not main or not isinstance(main.get("geometry"), dict):
        return
    fp = main["geometry"].get("footprint")
    if not isinstance(fp, dict):
        return
    fp = _sanitize_footprint(fp, size=main.get("size"), default_type=main["geometry"].get("shape"))
    main["geometry"]["footprint"] = fp
    if fp.get("type") == "composite_2d":
        main["geometry"]["shape"] = "composite_2d"
        if "semicircle" in json.dumps(fp, ensure_ascii=False).lower():
            main["subtype"] = "rectangular_stage_with_semicircular_apron"
    stage = scene.setdefault("stage", {})
    if isinstance(stage, dict):
        stage["footprint"] = deepcopy(fp)
        geom = stage.setdefault("geometry", {}) if isinstance(stage.get("geometry"), dict) else {}
        if not geom:
            stage["geometry"] = geom
        geom.setdefault("representation", "contour_extrusion")
        geom["shape"] = "composite_2d" if fp.get("type") == "composite_2d" else fp.get("type", geom.get("shape", "unknown"))
        geom["footprint"] = deepcopy(fp)
        if main.get("size", {}).get("h") and not geom.get("extrude_height"):
            geom["extrude_height"] = main["size"]["h"]

def sanitize_scene_json(data):
    if not isinstance(data, dict):
        return data
    data["schema_version"] = "0.4"
    scene = data.setdefault("scene", {})
    raw_stage = deepcopy(scene.get("stage")) if isinstance(scene.get("stage"), dict) else {}
    raw_objects = _as_list(scene.get("objects"))
    raw_groups = _as_list(scene.get("object_groups"))
    raw_relations = _as_list(scene.get("relations"))
    scene["coordinate_system"] = _sanitize_coordinate_system(scene.get("coordinate_system"))
    scene["stage"] = _sanitize_stage(scene.get("stage"))
    objects = [x for x in (_sanitize_object(o) for o in raw_objects) if x]
    has_platform_group = any(isinstance(g, dict) and str(g.get("category", "")).strip().lower().replace("-", "_").replace(" ", "_") == "platform" for g in raw_groups)
    recovered_stage = None if has_platform_group else _stage_object_candidate(raw_stage, objects, raw_relations)
    if recovered_stage:
        objects.insert(0, recovered_stage)
        # Keep enough floor/envelope for recovered stage plus upstage structures.
        try:
            scene["stage"]["width"] = max(float(scene["stage"].get("width", 12)), float(recovered_stage.get("size", {}).get("w", 6)) + 4.0)
            scene["stage"]["depth"] = max(float(scene["stage"].get("depth", 8)), float(recovered_stage.get("size", {}).get("d", 4)) + 3.0)
        except Exception:
            pass
    scene["objects"] = objects
    _sync_stage_footprint_from_main_stage(scene)
    scene["object_groups"] = [x for x in (_sanitize_group(g) for g in _as_list(scene.get("object_groups"))) if x]
    scene["relations"] = [x for x in (_sanitize_relation(r) for r in raw_relations) if x]
    scene["constraints"] = [x for x in (_sanitize_constraint(c) for c in _as_list(scene.get("constraints"))) if x]
    scene["cameras"] = [x for x in (_sanitize_camera(c, i) for i, c in enumerate(_as_list(scene.get("cameras")))) if x]
    scene["source_evidence"] = _sanitize_source_evidence_list(scene.get("source_evidence"))
    scene["validation"] = _sanitize_validation(scene.get("validation"))
    from decision_state import initialize
    initialize(scene)
    # Normalize top-level metadata only if present; schema allows metadata object fields.
    return data


# --------------------------------------------------------------------
class Validator:
    def __init__(self, scene):
        self.scene = scene
        self.stage = scene.get("stage", {})
        self.schema_version = scene.get("_schema_version", None)
        self.objects = scene.get("objects", [])
        self.by_id = {o["id"]: o for o in self.objects if "id" in o}
        self.expanded_group_map = {}
        self.filled = []     # audit trail of filled fields
        self.conflicts = []  # detected conflicts

    def record(self, obj_id, field, value, reason):
        self.filled.append({"object": obj_id, "field": field, "value": value, "reason": reason})


    def refresh_index(self):
        self.objects = self.scene.get("objects", [])
        self.by_id = {o["id"]: o for o in self.objects if "id" in o}

    def _clean_none(self, d):
        return {k: v for k, v in d.items() if v is not None}

    def _copy_json(self, value):
        return json.loads(json.dumps(value, ensure_ascii=False))

    def _source_views_from_evidence(self, evidence):
        views = []
        if not isinstance(evidence, list):
            return views
        for e in evidence:
            if not isinstance(e, dict):
                continue
            src = e.get("source")
            if src and src not in views:
                views.append(src)
        return views

    def _distribution_points(self, group):
        dist = group.get("distribution") or group.get("layout") or {}
        count = int(group.get("count", 0) or 0)
        dtype = dist.get("type")
        if count <= 0:
            return []
        raw_positions = dist.get("positions") or dist.get("points")
        if isinstance(raw_positions, list):
            pts = []
            heights = dist.get("heights") if isinstance(dist.get("heights"), list) else []
            proto_cat = (group.get("prototype") or {}).get("category", group.get("category"))
            for i, p in enumerate(raw_positions):
                if isinstance(p, list) and len(p) >= 2:
                    x, y = p[0], p[1]
                    # For platform groups, the third coordinate from VLM is often
                    # image-depth pollution.  Height belongs in size.h, not z.
                    z = 0.0 if proto_cat == "platform" else (p[2] if len(p) >= 3 else 0.0)
                    if proto_cat != "platform" and isinstance(z, (int, float)) and abs(float(z)) > 2.0:
                        z = 0.0
                    pts.append([x, y, z])
            if len(pts) >= count:
                return pts[:count]
            if pts:
                pts = pts + [pts[-1]] * (count - len(pts))
                return pts
        if dtype == "manual_points" and isinstance(dist.get("points"), list):
            pts = [p for p in dist.get("points", []) if isinstance(p, list) and len(p) == 3]
            if len(pts) >= count:
                return pts[:count]
            if pts:
                pts = pts + [pts[-1]] * (count - len(pts))
                return pts
        if dtype == "linear" and isinstance(dist.get("from"), list) and isinstance(dist.get("to"), list):
            a, b = dist["from"], dist["to"]
            if len(a) == 3 and len(b) == 3:
                if count == 1:
                    return [[round((a[i] + b[i]) / 2.0, 3) for i in range(3)]]
                return [[round(a[j] + (b[j] - a[j]) * i / (count - 1), 3) for j in range(3)] for i in range(count)]
        # Fallback: stack near stage center, validator can refine later.
        depth = self.stage.get("depth", 8.0)
        return [[0.0, round(depth * 0.5, 3), 0.0] for _ in range(count)]

    def expand_object_groups(self):
        """Expand v0.3 object_groups into ordinary objects for downstream
        validators and Blender generators. The original group is preserved; the
        expanded children get group_id and generated relations."""
        groups = self.scene.get("object_groups") or []
        if not groups:
            return
        relations = self.scene.setdefault("relations", [])
        existing_ids = {o.get("id") for o in self.objects}
        existing_rels = {(r.get("type"), r.get("subject"), r.get("target"))
                         for r in relations if isinstance(r, dict)}

        def add_rel(rtype, sub, tgt, source="validator", confidence=None):
            if not (rtype and sub and tgt):
                return
            key = (rtype, sub, tgt)
            if key in existing_rels:
                return
            rel = {"type": rtype, "subject": sub, "target": tgt, "source": source}
            if confidence is not None:
                rel["confidence"] = confidence
            relations.append(rel)
            existing_rels.add(key)
            self.record(sub, f"relation.{rtype}", tgt, "expanded_from_object_group")

        for group in groups:
            gid = group.get("id")
            count = int(group.get("count", 0) or 0)
            proto = group.get("prototype") or {}
            if not gid or count <= 0 or not proto:
                continue
            expanded_ids = list(group.get("expanded_object_ids") or [])
            while len(expanded_ids) < count:
                expanded_ids.append(f"{gid}_{len(expanded_ids) + 1}")
            pts = self._distribution_points(group)
            self.expanded_group_map[gid] = expanded_ids[:count]

            for i, oid in enumerate(expanded_ids[:count]):
                if oid in existing_ids:
                    continue
                obj = {
                    "id": oid,
                    "category": proto.get("category", "prop"),
                    "subtype": proto.get("subtype"),
                    "label": f"{proto.get('label', group.get('label', gid))} {i + 1}",
                    "group_id": gid,
                    "position": pts[i] if i < len(pts) else None,
                    "size": self._copy_json(proto.get("size")) if proto.get("size") else None,
                    "geometry": self._copy_json(proto.get("geometry")) if proto.get("geometry") else None,
                    "material": self._copy_json(proto.get("material")) if proto.get("material") else None,
                    "placement": self._copy_json(proto.get("placement")) if proto.get("placement") else None,
                    "attributes": self._copy_json(proto.get("attributes")) if proto.get("attributes") else None,
                    "asset": self._copy_json(proto.get("asset")) if proto.get("asset") else None,
                    "generation_hints": self._copy_json(proto.get("generation_hints")) if proto.get("generation_hints") else None,
                    "confidence": proto.get("confidence", group.get("confidence")),
                    "source": "multi_view",
                    "source_views": self._source_views_from_evidence(group.get("source_evidence")),
                    "source_evidence": self._copy_json(group.get("source_evidence")) if group.get("source_evidence") else None,
                }
                dist = group.get("distribution") or {}
                heights = dist.get("heights") if isinstance(dist.get("heights"), list) else []
                if i < len(heights) and isinstance(obj.get("size"), dict):
                    try:
                        obj["size"]["h"] = float(heights[i])
                        self.record(oid, "size.h", obj["size"]["h"], "expanded_from_group_layout_height")
                    except Exception:
                        pass
                obj = self._clean_none(obj)
                self.objects.append(obj)
                existing_ids.add(oid)
                self.record(oid, "object", gid, "expanded_from_object_group")

                pl = obj.get("placement") or {}
                if pl.get("mounted_on"):
                    add_rel("mounted_on", oid, pl["mounted_on"], source=obj.get("source", "validator"), confidence=obj.get("confidence"))
                if pl.get("target"):
                    # For lights, aimed_at + light_focus are both useful downstream.
                    add_rel("aimed_at", oid, pl["target"], source=obj.get("source", "validator"), confidence=obj.get("confidence"))
                    if obj.get("category") == "light":
                        add_rel("light_focus", oid, pl["target"], source=obj.get("source", "validator"), confidence=obj.get("confidence"))

            # Parse compact target_relation like "mounted_on:overhead_truss".
            target_relation = (group.get("distribution") or {}).get("target_relation")
            if isinstance(target_relation, str) and ":" in target_relation:
                rtype, tgt = target_relation.split(":", 1)
                for oid in expanded_ids[:count]:
                    add_rel(rtype, oid, tgt, source="multi_view", confidence=group.get("confidence"))

        # Expand relations that refer to a group id into equivalent child relations.
        original_relations = list(relations)
        for r in original_relations:
            subject = r.get("subject")
            if subject not in self.expanded_group_map:
                continue
            for oid in self.expanded_group_map[subject]:
                add_rel(r.get("type"), oid, r.get("target"), source=r.get("source", "multi_view"), confidence=r.get("confidence"))
        self.scene["objects"] = self.objects
        self.refresh_index()

    def normalize_placement_relations(self):
        """Move relation-like placement keys into scene.relations.

        This repairs common v0.2 raw-model slips such as
        placement.behind, placement.above, and placement.on_top_of. It also
        removes unknown placement keys so the filled scene can pass schema
        validation instead of failing on harmless relation-placement confusion.
        """
        relations = self.scene.setdefault("relations", [])
        existing = {(r.get("type"), r.get("subject"), r.get("target"))
                    for r in relations if isinstance(r, dict)}

        def add_relation(rtype, subject, target, source, original_key=None):
            if not subject or not isinstance(target, str) or not target:
                return
            key = (rtype, subject, target)
            if key in existing:
                return
            rel = {"type": rtype, "subject": subject, "target": target, "source": source or DEFAULT_SOURCE}
            if original_key == "above":
                rel["params"] = {"normalized_from": "placement.above", "vertical_relation": "above"}
            relations.append(rel)
            existing.add(key)
            self.record(subject, f"placement.{original_key or rtype}", target, f"moved_to_relation:{rtype}")

        for o in self.objects:
            pl = o.get("placement")
            if not isinstance(pl, dict):
                continue
            oid = o.get("id")
            for k in list(pl.keys()):
                if k in PLACEMENT_ALLOWED_KEYS:
                    continue
                value = pl.pop(k)
                rtype = ("above" if (self.schema_version == "0.3" and k == "above") else PLACEMENT_RELATION_KEYS.get(k))
                if not rtype:
                    self.record(oid, f"placement.{k}", value, "removed_unknown_placement_key")
                    continue
                targets = value if isinstance(value, list) else [value]
                for target in targets:
                    add_relation(rtype, oid, target, o.get("source", DEFAULT_SOURCE), original_key=k)

    # ---- layout recovery helpers ------------------------------------
    def _primary_stage_object(self):
        platforms = [o for o in self.objects if o.get("category") == "platform"]
        if not platforms:
            return None
        def area(o):
            s = o.get("size") or {}
            return float(s.get("w", 0) or 0) * float(s.get("d", 0) or 0)
        for o in platforms:
            if o.get("id") == "main_stage":
                return o
        return max(platforms, key=area)

    def _infer_side_from_text(self, o):
        text = " ".join(str(o.get(k, "")) for k in ("id", "label", "subtype", "description")).lower()
        if any(x in text for x in ("stage_left", "left", "左")):
            return "left"
        if any(x in text for x in ("stage_right", "right", "右")):
            return "right"
        if any(x in text for x in ("center", "centre", "middle", "中央", "中间")):
            return "center"
        if any(x in text for x in ("front", "downstage", "前")):
            return "downstage"
        if any(x in text for x in ("back", "upstage", "后")):
            return "upstage"
        return None

    def _is_monitor_wedge(self, o):
        text = " ".join(str(o.get(k, "")) for k in ("id", "label", "subtype", "description")).lower()
        return o.get("category") == "audio" and ("monitor" in text or "wedge" in text or "返听" in text or "监听" in text)

    def _semantic_text(self, o):
        return " ".join(str(o.get(k, "")) for k in ("id", "label", "category", "subtype", "description")).lower().replace("_", " ").replace("-", " ")

    def _is_line_array_audio(self, o):
        text = self._semantic_text(o)
        if o.get("category") != "audio":
            return False
        if self._is_monitor_wedge(o):
            return False
        return ("line array" in text or "linearray" in text or "speaker array" in text or "side speaker" in text or "array" in text or "线阵" in text or "音响阵列" in text)

    def _is_side_wall_or_column(self, o):
        text = self._semantic_text(o)
        if o.get("category") not in ("set_piece", "wall", "column", "arch", "scenic_background", "decor"):
            return False
        return any(k in text for k in ("proscenium", "side wall", "sidewall", "side column", "column", "pillar", "wall", "侧墙", "侧柱", "柱"))

    def _is_curtain_like(self, o):
        text = self._semantic_text(o)
        return o.get("category") in ("curtain", "scenic_background") or "curtain" in text or "幕布" in text

    def _is_overhead_truss(self, o):
        text = self._semantic_text(o)
        pl = o.get("placement") or {}
        return o.get("category") == "truss" and (pl.get("zone") == "overhead" or "overhead" in text or "lighting" in text or "truss" in text)

    def _stage_profile(self, stage=None):
        """Return derived local anchors for the main platform.

        The result is generic: it turns a stage footprint into named layout
        anchors without hard-coding a particular scene.  Composite
        rectangle+semicircle stages expose transition anchors, curved-apron
        anchors and rectangular-platform bounds; rectangular/circular stages get
        simpler front/side anchors.
        """
        stage = stage or self._primary_stage_object()
        if not (stage and stage.get("position") and stage.get("size")):
            return None
        sx, sy, sz = [float(x) for x in stage.get("position", [0,0,0])[:3]]
        ss = stage.get("size") or {}
        sw = float(ss.get("w", 4.0) or 4.0)
        sd = float(ss.get("d", 3.0) or 3.0)
        sh = float(ss.get("h", 0.3) or 0.3)
        geom = stage.get("geometry") or {}
        fp = geom.get("footprint") if isinstance(geom.get("footprint"), dict) else {}
        ftype = str(fp.get("type") or geom.get("shape") or stage.get("subtype") or "").lower()
        prof = {
            "kind": ftype or "unknown",
            "x": sx, "y": sy, "z": sz, "w": sw, "d": sd, "h": sh, "top_z": sz + sh,
            "front_y": sy - sd / 2.0, "back_y": sy + sd / 2.0,
            "left_x": sx - sw / 2.0, "right_x": sx + sw / 2.0,
            "is_rect_semicircle": False,
        }
        parts = fp.get("parts") if isinstance(fp, dict) else None
        if isinstance(parts, list):
            def ptype(p):
                return str((p or {}).get("type") or (p or {}).get("shape") or "").lower().replace("-", "_")
            rect = next((p for p in parts if ptype(p) in ("rectangle", "rect", "box")), None)
            semi = next((p for p in parts if ptype(p) in ("semicircle", "semi_circle", "semicircular_apron", "semi_circular_apron")), None)
            if rect and semi:
                rw = float(rect.get("w", sw) or sw)
                rd = float(rect.get("d", sd) or sd)
                r = float(semi.get("radius", min(rw, rd) / 2.0) or min(rw, rd) / 2.0)
                side = str(semi.get("side", "downstage")).lower()
                if side in ("downstage", "front", "-y"):
                    rect_front_y = sy - rd / 2.0
                    apron_front_y = rect_front_y - r
                    prof.update({
                        "kind": "rectangle_with_semicircular_apron",
                        "is_rect_semicircle": True,
                        "rect_w": rw, "rect_d": rd, "apron_radius": r,
                        "rect_front_y": rect_front_y,
                        "rect_back_y": sy + rd / 2.0,
                        "apron_front_y": apron_front_y,
                        "transition_y": rect_front_y,
                        "transition_left_x": sx - r,
                        "transition_right_x": sx + r,
                        "rect_left_x": sx - rw / 2.0,
                        "rect_right_x": sx + rw / 2.0,
                        "front_y": apron_front_y,
                        "back_y": sy + rd / 2.0,
                        "left_x": sx - rw / 2.0,
                        "right_x": sx + rw / 2.0,
                    })
        return prof

    def _arc_y_inside_downstage_apron(self, profile, x, inset=0.25, half_w=0.0, half_d=0.0):
        """Y position safely inside a downstage semicircular apron at world x.

        Earlier versions used only the object center when following a curved
        apron. That can leave the left/right front corners outside the platform
        even when the center point is inside. This helper is footprint-aware: it
        clamps x by object half width and computes the safest front edge across
        the left/right corners, then adds half depth plus inset. It is generic
        for any small object resting on a semicircular apron.
        """
        r = float(profile.get("apron_radius", 0.0) or 0.0)
        if r <= 0:
            return profile.get("front_y", profile["y"]) + inset + half_d
        safe_half_w = max(0.0, float(half_w or 0.0))
        safe_half_d = max(0.0, float(half_d or 0.0))
        # Ensure the full object width can fit inside the arc chord.
        local_x = float(x) - profile["x"]
        limit = max(0.01, r - safe_half_w - 0.06)
        local_x = max(-limit, min(limit, local_x))
        x = profile["x"] + local_x

        # Compute the most upstage/front-edge requirement among both front
        # corners; this keeps the whole rectangular footprint supported.
        edge_candidates = []
        for cx in (local_x - safe_half_w, local_x + safe_half_w, local_x):
            cx = max(-r + 0.01, min(r - 0.01, cx))
            edge_candidates.append(profile["rect_front_y"] - math.sqrt(max(0.0, r * r - cx * cx)))
        required_y = max(edge_candidates) + safe_half_d + inset
        return required_y

    def _allows_visual_overhang(self, o):
        """Return True when visual/user evidence explicitly allows an object to cross a support boundary."""
        blob = self._semantic_text(o) + " " + json.dumps(o.get("source_evidence", []), ensure_ascii=False).lower()
        return any(k in blob for k in (
            "overhang", "over edge", "extends past", "outside platform", "crosses edge",
            "悬挑", "伸出", "超出舞台", "跨出台面", "跨过边缘"
        ))

    def _profile_contains_rect_semicircle_bbox(self, profile, x, y, half_w, half_d, margin=0.04):
        """Conservative containment test for an axis-aligned small object on a rect+semicircle platform."""
        if not profile or not profile.get("is_rect_semicircle"):
            return True
        r = float(profile.get("apron_radius", 0.0) or 0.0)
        cx0 = float(profile["x"])
        rect_left = float(profile.get("rect_left_x", profile["left_x"])) + margin
        rect_right = float(profile.get("rect_right_x", profile["right_x"])) - margin
        rect_front = float(profile["rect_front_y"]) + margin
        rect_back = float(profile.get("rect_back_y", profile["back_y"])) - margin
        transition_y = float(profile["rect_front_y"])
        half_w = max(0.0, float(half_w or 0.0))
        half_d = max(0.0, float(half_d or 0.0))

        corners = [(x-half_w, y-half_d), (x+half_w, y-half_d), (x-half_w, y+half_d), (x+half_w, y+half_d)]
        for px, py in corners:
            in_rect = (rect_left <= px <= rect_right and rect_front <= py <= rect_back)
            dx = px - cx0
            in_apron = (py <= transition_y + 1e-6 and dx*dx + (py - transition_y) * (py - transition_y) <= (r - margin) * (r - margin) + 1e-6)
            if not (in_rect or in_apron):
                return False
        return True

    def _clamp_to_rect_semicircle_profile(self, profile, x, y, half_w, half_d, preferred_zone="downstage_apron"):
        """Project a small object center to a safe point inside a rect+semicircle profile."""
        r = float(profile.get("apron_radius", 0.0) or 0.0)
        if r <= 0:
            return x, y
        half_w = max(0.0, float(half_w or 0.0))
        half_d = max(0.0, float(half_d or 0.0))
        margin = max(0.06, min(0.18, max(half_w, half_d) * 0.35))
        # Clamp horizontally so both side corners can fit.
        limit = max(0.01, r - half_w - margin)
        x = profile["x"] + max(-limit, min(limit, float(x) - profile["x"]))
        # For downstage/apron equipment, keep the object just inside the front
        # arc using its full bounding footprint, not only its center.
        if preferred_zone in ("downstage_apron", "downstage", "front"):
            y_min = self._arc_y_inside_downstage_apron(profile, x, inset=margin, half_w=half_w, half_d=half_d)
            y = max(float(y), y_min)
            # Avoid pushing it so far upstage that it looks detached from the apron.
            y = min(y, profile["rect_front_y"] - half_d - margin * 0.25)
        # If still outside because of rough dimensions, move further toward
        # stage center until the whole box is contained.
        for _ in range(12):
            if self._profile_contains_rect_semicircle_bbox(profile, x, y, half_w, half_d, margin=margin * 0.4):
                break
            x = profile["x"] + (x - profile["x"]) * 0.88
            y = y + 0.08
        return round(x, 3), round(y, 3)

    def _add_relation_once(self, rtype, subject, target, source="validator", confidence=0.65):
        if not subject or not target:
            return
        rels = self.scene.setdefault("relations", [])
        key = (rtype, subject, target)
        for r in rels:
            if (r.get("type"), r.get("subject"), r.get("target")) == key:
                return
        rels.append({"type": rtype, "subject": subject, "target": target, "source": source, "confidence": confidence})
        self.record(subject, f"relation.{rtype}", target, "layout_recovery_relation")

    def recover_missing_main_stage(self):
        """If the extractor omitted a platform object, synthesize a conservative
        main_stage so attached stairs, monitors, curtains and lights have an anchor.
        This is generic recovery, not a scene-specific hard-code.
        """
        if self._primary_stage_object():
            return
        if any(isinstance(g, dict) and g.get("category") == "platform" for g in self.scene.get("object_groups", []) or []):
            # MULTIVIEW_STRUCTURE_NORMALIZATION_V2: compact platform groups are
            # real stage surfaces; do not synthesize an extra circular/rectangular
            # main_stage before group expansion.
            return
        # Only recover when there is evidence that stage hardware needs a stage anchor.
        has_stage_dependents = any(o.get("category") in ("stairs", "audio", "curtain", "scenic_background", "truss", "light") for o in self.objects)
        rel_mentions = json.dumps(self.scene.get("relations", []), ensure_ascii=False).lower()
        if not has_stage_dependents and "main_stage" not in rel_mentions:
            return
        W = float(self.stage.get("width", 12) or 12)
        D = float(self.stage.get("depth", 8) or 8)
        w = min(6.0, max(3.5, W * 0.55))
        d = min(5.0, max(3.0, D * 0.45))
        obj = {
            "id": "main_stage",
            "category": "platform",
            "subtype": "main_stage",
            "label": "recovered main stage platform",
            "position": [0.0, round(D * 0.42, 3), 0.0],
            "size": {"w": round(w, 3), "d": round(d, 3), "h": 0.3},
            "geometry": {"representation": "contour_extrusion", "shape": "rectangle", "footprint": {"type": "rectangle", "w": round(w, 3), "d": round(d, 3)}, "extrude_height": 0.3, "coordinate_mode": "bottom_center"},
            "material": {"type": "wood", "color_name": "warm_wood"},
            "placement": {"zone": "center", "anchor": "bottom_center"},
            "source": "validator",
            "confidence": 0.45,
            "source_views": ["inferred"],
            "source_evidence": [{"source": "validator", "supports": ["existence", "layout_anchor"], "confidence": 0.45, "notes": "Recovered because dependent stage objects existed but no platform object was present."}],
        }
        self.objects.insert(0, obj)
        self.scene["objects"] = self.objects
        self.refresh_index()
        self.record("main_stage", "object", "platform", "recovered_missing_platform_anchor")

    def recover_layout_semantics(self):
        """Infer missing side/distribution semantics from id/label/subtype and
        object counts. This repairs common VLM raw JSON where objects are named
        left/right/center but their placement fields are absent.
        """
        stage = self._primary_stage_object()
        if not stage:
            return
        sid = stage.get("id")
        for o in self.objects:
            pl = o.setdefault("placement", {}) if isinstance(o.get("placement"), dict) else {}
            if not isinstance(o.get("placement"), dict):
                o["placement"] = pl
            if o.get("category") == "stairs":
                side = pl.get("side") if pl.get("side") in PLACEMENT_SIDE_VALUES else None
                inferred = side or self._infer_side_from_text(o)
                if inferred:
                    if pl.get("side") != inferred:
                        pl["side"] = inferred
                        self.record(o["id"], "placement.side", inferred, "inferred_from_id_label")
                if not pl.get("attached_to"):
                    pl["attached_to"] = sid
                    self.record(o["id"], "placement.attached_to", sid, "inferred_stage_anchor")
                    self._add_relation_once("attached_to", o["id"], sid, source="validator")
            elif self._is_monitor_wedge(o):
                if not pl.get("target"):
                    pl["target"] = sid
                if not pl.get("zone"):
                    pl["zone"] = "downstage"
                self._add_relation_once("on_top_of", o["id"], sid, source="validator")
        self.refresh_index()

    def distribute_monitor_wedges(self):
        stage = self._primary_stage_object()
        if not stage or not (stage.get("position") and stage.get("size")):
            return
        monitors = [o for o in self.objects if self._is_monitor_wedge(o) and o.get("size")]
        if len(monitors) < 2:
            return
        # If positions are already deliberately distinct, do not override. If
        # they all stack or were defaulted, distribute left/center/right.  The
        # composite-stage pass below may still upgrade a simple line into an
        # apron/arc/triangle pattern when the stage footprint supports it.
        pts = [(round((o.get("position") or [0,0,0])[0], 2), round((o.get("position") or [0,0,0])[1], 2)) for o in monitors]
        unique_pts = set(pts)
        if len(unique_pts) > max(1, len(monitors) // 2):
            return
        sx, sy, sz = stage["position"]
        ss = stage["size"]
        sw, sd, sh = ss.get("w", 4.0), ss.get("d", 3.0), ss.get("h", 0.3)
        top_z = round(sz + sh, 3)
        n = len(monitors)
        # Spread within the platform width, not all the way to the edge.
        span = min(sw * 0.65, max(0.8, (n - 1) * 1.2))
        if n == 1:
            xs = [sx]
        else:
            xs = [sx - span / 2.0 + span * i / (n - 1) for i in range(n)]
        geom = stage.get("geometry") or {}
        shape_blob = json.dumps(geom, ensure_ascii=False).lower() + " " + str(stage.get("subtype", "")).lower()
        for i, o in enumerate(monitors):
            x = xs[i]
            od = (o.get("size") or {}).get("d", 0.4)
            # Curved/circular platform front edge: y_edge varies by x.
            if "circle" in shape_blob or "round" in shape_blob or "圆" in shape_blob:
                r = min(sw, sd) / 2.0
                local_x = max(-r + 0.1, min(r - 0.1, x - sx))
                y_edge = sy - math.sqrt(max(0.0, r * r - local_x * local_x))
                y = y_edge + od / 2.0 + 0.12
            else:
                y = sy - sd / 2.0 + od / 2.0 + 0.12
            o["position"] = [round(x, 3), round(y, 3), top_z]
            side = "center"
            if i == 0 and n > 1:
                side = "left"
            elif i == n - 1 and n > 1:
                side = "right"
            pl = o.setdefault("placement", {})
            pl["side"] = side
            pl["zone"] = "downstage"
            pl["target"] = stage["id"]
            self.record(o["id"], "position", o["position"], "distributed_monitor_wedge_array")
            self.record(o["id"], "placement.side", side, "distributed_monitor_wedge_array")

    def recover_composite_stage_layout_anchors(self):
        """Recover local anchors and layout patterns for composite stages.

        This is a generic pass for stages whose footprint exposes a curved
        downstage apron plus a rectangular/upstage body.  It does not hard-code
        one sample scene; it derives anchors from the stage footprint:
        - stairs attach to left/right transition zones between rectangle/apron;
        - monitor wedges use arc/triangle placement on the apron region;
        - side speakers attach to side wall/column front faces or truss ends;
        - truss, curtain, side walls keep basic clearance layers.
        """
        stage = self._primary_stage_object()
        prof = self._stage_profile(stage)
        if not (stage and prof):
            return
        top_z = round(prof["top_z"], 3)

        # 1) Side walls/columns flank the curtain and rest on the platform top.
        curtains = [o for o in self.objects if self._is_curtain_like(o) and o.get("position") and o.get("size")]
        curtain = max(curtains, key=lambda o: (o.get("size") or {}).get("w", 0), default=None)
        side_walls = [o for o in self.objects if self._is_side_wall_or_column(o) and o.get("size")]
        if curtain and side_walls:
            cx, cy, cz = curtain.get("position", [prof["x"], prof["back_y"], top_z])[:3]
            cs = curtain.get("size") or {}
            cw = float(cs.get("w", prof["w"]) or prof["w"])
            # Keep proscenium/side structures just outside curtain span rather
            # than overlapping the curtain body.  This is a general flank rule.
            for wall in side_walls:
                side = (wall.get("placement") or {}).get("side") or self._infer_side_from_text(wall) or "center"
                ws = wall.get("size") or {}
                ww = float(ws.get("w", 0.6) or 0.6)
                if side == "left":
                    x = float(cx) - cw / 2.0 - ww / 2.0 - 0.05
                elif side == "right":
                    x = float(cx) + cw / 2.0 + ww / 2.0 + 0.05
                else:
                    continue
                wall["position"] = [round(x, 3), round(float(cy), 3), top_z]
                pl = wall.setdefault("placement", {})
                pl["side"] = side
                pl["zone"] = "upstage_side"
                self.record(wall["id"], "position", wall["position"], "side_wall_flanks_curtain_clearance")

        # 2) Stairs attach to transition zones of composite stages, not to the
        # far left/right side of the whole rectangular bounding box.
        if prof.get("is_rect_semicircle"):
            for stair in [o for o in self.objects if o.get("category") == "stairs" and o.get("size")]:
                side = (stair.get("placement") or {}).get("side") or self._infer_side_from_text(stair)
                if side not in ("left", "right"):
                    continue
                ss = stair.get("size") or {}
                sw = float(ss.get("w", 1.2) or 1.2)
                sd = float(ss.get("d", 1.0) or 1.0)
                # Place just outside the apron shoulder.  The shoulder is the
                # semantic connection between the rectangular deck and curved
                # downstage apron.
                shoulder_x = prof["transition_left_x"] if side == "left" else prof["transition_right_x"]
                sign = -1 if side == "left" else 1
                x = shoulder_x + sign * (sw / 2.0 + 0.08)
                y = prof["transition_y"] - sd * 0.35
                stair["position"] = [round(x, 3), round(y, 3), 0.0]
                pl = stair.setdefault("placement", {})
                pl["attached_to"] = stage["id"]
                pl["side"] = side
                pl["zone"] = "downstage_transition"
                self._add_relation_once("attached_to", stair["id"], stage["id"], source="validator")
                self.record(stair["id"], "position", stair["position"], "composite_stage_transition_stairs")

        # 3) Monitor wedges on a curved apron should follow the apron, not a
        # single straight line.  For three monitors use a shallow triangular /
        # fan layout: center at the apron front, left/right along the shoulders.
        monitors = [o for o in self.objects if self._is_monitor_wedge(o) and o.get("size")]
        if prof.get("is_rect_semicircle") and len(monitors) >= 2:
            def side_key(o):
                side = (o.get("placement") or {}).get("side") or self._infer_side_from_text(o) or "center"
                return {"left": 0, "center": 1, "right": 2}.get(side, 1)
            monitors = sorted(monitors, key=side_key)
            n = len(monitors)
            r = float(prof.get("apron_radius", min(prof["w"], prof["d"]) / 2.0) or 1.0)
            if n == 3:
                desired = [
                    ("left", prof["x"] - r * 0.62),
                    ("center", prof["x"]),
                    ("right", prof["x"] + r * 0.62),
                ]
            else:
                span = min(r * 1.65, max(0.8, (n - 1) * 0.95))
                desired = []
                for i in range(n):
                    side = "center"
                    if i == 0:
                        side = "left"
                    elif i == n - 1:
                        side = "right"
                    desired.append((side, prof["x"] - span / 2.0 + span * i / max(1, n - 1)))
            for o, (side, x) in zip(monitors, desired):
                osize = o.get("size") or {}
                ow = float(osize.get("w", 0.4) or 0.4)
                od = float(osize.get("d", 0.4) or 0.4)
                inset = max(0.08, min(0.18, max(ow, od) * 0.28))
                y = self._arc_y_inside_downstage_apron(prof, x, inset=inset, half_w=ow/2.0, half_d=od/2.0)
                x, y = self._clamp_to_rect_semicircle_profile(prof, x, y, ow/2.0, od/2.0, preferred_zone="downstage_apron")
                o["position"] = [round(x, 3), round(y, 3), top_z]
                pl = o.setdefault("placement", {})
                pl["side"] = side
                pl["zone"] = "downstage_apron"
                pl["target"] = stage["id"]
                pl["notes"] = "layout_pattern: apron_triangle_or_arc; containment: keep_footprint_inside_support_surface unless visual_overhang"
                gh = o.setdefault("generation_hints", {})
                gh["surface_containment"] = "inside_support_footprint"
                self._add_relation_once("on_top_of", o["id"], stage["id"], source="validator")
                self.record(o["id"], "position", o["position"], "composite_stage_monitor_apron_pattern_contained")

        # 4) Truss should sit in front of the curtain/back wall layer while
        # remaining overhead.  This avoids the truss being embedded into scenic
        # background geometry.
        if curtain:
            cy = float(curtain.get("position", [0, prof["back_y"], 0])[1])
            cd = float((curtain.get("size") or {}).get("d", 0.12) or 0.12)
            ch = float((curtain.get("size") or {}).get("h", 4.0) or 4.0)
            cz = float((curtain.get("position") or [0, 0, top_z])[2])
            for truss in [o for o in self.objects if self._is_overhead_truss(o) and o.get("size")]:
                ts = truss.get("size") or {}
                td = float(ts.get("d", 0.35) or 0.35)
                th = float(ts.get("h", 0.35) or 0.35)
                pos = truss.setdefault("position", [prof["x"], cy - 0.4, top_z + ch + 0.4])
                pos[0] = round(prof["x"], 3)
                pos[1] = round(cy - cd / 2.0 - td / 2.0 - 0.18, 3)
                pos[2] = round(max(float(pos[2]), cz + ch + th / 2.0 + 0.05), 3)
                truss["position"] = pos
                self.record(truss["id"], "position", truss["position"], "overhead_truss_front_of_curtain_clearance")

        # 5) Line-array / side-speaker audio attaches to side wall front faces
        # or, if there are no walls, to truss end zones.  This separates flown /
        # wall-mounted audio from floor monitor wedges.
        side_walls_by_side = {}
        for wall in side_walls:
            side = (wall.get("placement") or {}).get("side") or self._infer_side_from_text(wall)
            if side in ("left", "right"):
                side_walls_by_side[side] = wall
        trusses = [o for o in self.objects if self._is_overhead_truss(o) and o.get("position") and o.get("size")]
        truss = max(trusses, key=lambda o: (o.get("size") or {}).get("w", 0), default=None)
        for spk in [o for o in self.objects if self._is_line_array_audio(o) and o.get("size")]:
            side = (spk.get("placement") or {}).get("side") or self._infer_side_from_text(spk)
            if side not in ("left", "right"):
                side = "left" if "left" in spk.get("id", "").lower() else "right" if "right" in spk.get("id", "").lower() else "center"
            ss = spk.get("size") or {}
            sw = float(ss.get("w", 0.5) or 0.5)
            sd = float(ss.get("d", 0.3) or 0.3)
            sh = float(ss.get("h", 2.0) or 2.0)
            if side in side_walls_by_side:
                wall = side_walls_by_side[side]
                wx, wy, wz = wall.get("position", [prof["x"], prof["back_y"], top_z])[:3]
                ws = wall.get("size") or {}
                wd = float(ws.get("d", 0.5) or 0.5)
                wh = float(ws.get("h", 3.0) or 3.0)
                x = float(wx)
                y = float(wy) - wd / 2.0 - sd / 2.0 - 0.08
                z = float(wz) + max(0.35, wh * 0.43)
                spk["position"] = [round(x, 3), round(y, 3), round(z, 3)]
                pl = spk.setdefault("placement", {})
                pl["side"] = side
                pl["zone"] = "side_wall_front"
                pl["attached_to"] = wall["id"]
                self._add_relation_once("attached_to", spk["id"], wall["id"], source="validator")
                self.record(spk["id"], "position", spk["position"], "side_speaker_on_side_wall_front")
            elif truss:
                tx, ty, tz = truss.get("position", [prof["x"], prof["back_y"], top_z + 3])[:3]
                tw = float((truss.get("size") or {}).get("w", prof["w"]) or prof["w"])
                sign = -1 if side == "left" else 1 if side == "right" else 0
                spk["position"] = [round(float(tx) + sign * tw * 0.46, 3), round(float(ty), 3), round(float(tz) - sh - 0.05, 3)]
                pl = spk.setdefault("placement", {})
                pl["side"] = side
                pl["zone"] = "truss_end_hung"
                pl["mounted_on"] = truss["id"]
                self._add_relation_once("mounted_on", spk["id"], truss["id"], source="validator")
                self.record(spk["id"], "position", spk["position"], "side_speaker_hung_from_truss_end")

    # ---- helpers -----------------------------------------------------
    def default_size(self, o):
        label = (o.get("label") or "").strip().lower()
        if label in DEFAULTS_BY_LABEL:
            return dict(DEFAULTS_BY_LABEL[label]), f"default_for_label:{label}"
        cat = o.get("category")
        if cat in DEFAULTS_BY_CATEGORY:
            return dict(DEFAULTS_BY_CATEGORY[cat]), f"default_for_category:{cat}"
        return {"w": 1.0, "d": 1.0, "h": 1.0}, "default_fallback"

    # ---- fill passes -------------------------------------------------
    def fill_coordinate_system(self):
        if "coordinate_system" not in self.scene:
            self.scene["coordinate_system"] = {
                "units": "m", "angle_units": "deg",
                "origin": "downstage_center_floor",
                "x": "audience_right", "y": "upstage_depth", "z": "up",
            }
            self.record("(scene)", "coordinate_system", "default convention", "ensured_present")

    def fill_sizes(self):
        for o in self.objects:
            if o.get("category") == "light":
                continue  # lights carry no size
            base, reason = self.default_size(o)
            sr = o.get("size_range") or {}
            size = o.get("size") or {}
            changed = False
            for axis in ("w", "d", "h"):
                if size.get(axis) is not None:
                    continue
                if axis in sr and len(sr[axis]) == 2:  # collapse range -> midpoint
                    mid = (sr[axis][0] + sr[axis][1]) / 2.0
                    size[axis] = mid
                    self.record(o["id"], f"size.{axis}", mid,
                                f"midpoint_of_size_range:{sr[axis]}")
                else:
                    size[axis] = base[axis]
                    self.record(o["id"], f"size.{axis}", base[axis], reason)
                changed = True
            if changed or "size" not in o:
                o["size"] = size

    def fill_positions(self):
        # iterative: resolve children once their parent has a position+size
        def resolved(o):
            return o.get("position") is not None
        for _ in range(len(self.objects) + 1):
            progressed = False
            for o in self.objects:
                if resolved(o):
                    continue
                pid = o.get("parent")
                parent = self.by_id.get(pid) if pid else None
                if parent and resolved(parent) and parent.get("size"):
                    px, py, pz = parent["position"]
                    top = pz + parent["size"].get("h", 0.0)
                    o["position"] = [px, py, top]  # parent top-center
                    self.record(o["id"], "position", [px, py, top],
                                f"on_top_of_parent:{pid}")
                    progressed = True
            if not progressed:
                break

        # anything still unplaced -> a category/subtype/placement-aware default
        width = self.stage.get("width", 12.0)
        depth = self.stage.get("depth", 8.0)
        clearance = self.stage.get("height_clearance", 6.0)
        for o in self.objects:
            if resolved(o):
                continue
            cat = o.get("category")
            subtype = (o.get("subtype") or "").lower()
            geom = o.get("geometry") or {}
            place = o.get("placement") or {}
            size = o.get("size") or {}
            w, d, h = size.get("w", 1.0), size.get("d", 1.0), size.get("h", 1.0)
            zone = place.get("zone")
            text = self._semantic_text(o) if hasattr(self, "_semantic_text") else " ".join(str(o.get(k, "")) for k in ("id", "label", "subtype", "description")).lower()
            if cat == "screen":
                pos = [0.0, max(0.0, depth - d / 2.0), 0.0]
                reason = "default_screen_upstage"
            elif cat in ("furniture", "decor"):
                # Array/group furniture from VLM output should not collapse to
                # dead center.  Use coarse semantic zones as a first stable
                # layout fallback; exact expansion is left to generator assets.
                if "right" in text or place.get("side") == "right":
                    pos = [round(width * 0.28, 3), round(depth * 0.55, 3), 0.0]
                    reason = "semantic_right_furniture_zone"
                elif "left" in text or place.get("side") == "left":
                    pos = [round(-width * 0.28, 3), round(depth * 0.55, 3), 0.0]
                    reason = "semantic_left_furniture_zone"
                elif any(k in text for k in ("stool", "chair", "seat", "audience", "观众", "凳", "椅")):
                    pos = [0.0, round(depth * 0.45, 3), 0.0]
                    reason = "semantic_furniture_floor_zone"
                else:
                    pos = [0.0, round(depth * 0.5, 3), 0.0]
                    reason = "default_furniture_stage_zone"
            elif cat in ("scenic_background", "curtain", "wall") :
                y = max(0.0, depth - d / 2.0)
                z = h / 2.0 if geom.get("coordinate_mode") == "center" or place.get("anchor") == "center" else 0.0
                pos = [0.0, y, round(z, 3)]
                reason = "default_scenic_background_upstage"
            elif cat == "truss":
                pos = [0.0, round(depth * 0.62, 3), round(clearance * 0.80, 3)]
                reason = "default_truss_overhead"
            elif cat == "light":
                if any(k in text for k in ("overhead", "array", "grid", "truss", "ceiling", "天花", "灯阵")) or place.get("zone") == "overhead":
                    pos = [0.0, round(depth * 0.55, 3), round(clearance * 0.78, 3)]
                    reason = "semantic_overhead_light_array"
                else:
                    pos = [0.0, round(depth * 0.50, 3), round(clearance * 0.75, 3)]
                    reason = "default_light_overhead"
            elif cat == "stairs":
                pos = [0.0, max(0.0, depth * 0.25), 0.0]
                reason = "default_stairs_downstage"
            elif cat == "audio":
                if "right" in text or place.get("side") == "right":
                    pos = [round(width * 0.42, 3), round(depth * 0.50, 3), 0.0]
                    reason = "semantic_right_audio_zone"
                elif "left" in text or place.get("side") == "left":
                    pos = [round(-width * 0.42, 3), round(depth * 0.50, 3), 0.0]
                    reason = "semantic_left_audio_zone"
                else:
                    y = depth * 0.35 if zone not in ("upstage", "upstage_left", "upstage_right") else depth * 0.65
                    pos = [0.0, round(y, 3), 0.0]
                    reason = "default_audio_stage_floor"
            else:
                pos = [0.0, depth / 2.0, 0.0]
                reason = "default_stage_center"
            o["position"] = pos
            self.record(o["id"], "position", pos, reason)

    def apply_placement_semantics(self):
        """Use v0.2 placement/relation fields to improve coordinates.

        The extractor now provides attached_to/mounted_on/target/relations. This
        pass turns those qualitative constraints into practical coordinates for
        stairs, backdrops, trusses, and truss-mounted lights.
        """
        rels = self.scene.get("relations", [])
        relation_targets = {}
        for r in rels:
            if r.get("subject") and r.get("target"):
                relation_targets.setdefault((r.get("type"), r.get("subject")), []).append(r.get("target"))

        # Stairs attached to the sides/front/back of a platform.
        for o in self.objects:
            if o.get("category") != "stairs" or not o.get("position") or not o.get("size"):
                continue
            pl = o.get("placement") or {}
            tgt_id = pl.get("attached_to") or next(iter(relation_targets.get(("attached_to", o["id"]), [])), None)
            tgt = self.by_id.get(tgt_id) if tgt_id else None
            if not (tgt and tgt.get("position") and tgt.get("size")):
                continue
            tx, ty, tz = tgt["position"]
            tw, td, th = tgt["size"].get("w", 0), tgt["size"].get("d", 0), tgt["size"].get("h", 0)
            sw, sd = o["size"].get("w", 1), o["size"].get("d", 1)
            side = (pl.get("side") or "front").lower()
            if "left" in side:
                pos = [tx - tw / 2.0 - sw / 2.0, ty, 0.0]
            elif "right" in side:
                pos = [tx + tw / 2.0 + sw / 2.0, ty, 0.0]
            elif side in ("back", "upstage") or "upstage" in side:
                pos = [tx, ty + td / 2.0 + sd / 2.0, 0.0]
            else:
                pos = [tx, ty - td / 2.0 - sd / 2.0, 0.0]
            o["position"] = [round(pos[0], 3), round(pos[1], 3), round(pos[2], 3)]
            self.record(o["id"], "position", o["position"], f"attached_to:{tgt_id}:{side}")

        # Backdrops/curtains behind the main stage.
        for r in rels:
            if r.get("type") != "behind":
                continue
            sub = self.by_id.get(r.get("subject"))
            tgt = self.by_id.get(r.get("target"))
            if not (sub and tgt and sub.get("position") and sub.get("size") and tgt.get("position") and tgt.get("size")):
                continue
            if sub.get("category") not in ("scenic_background", "screen", "prop"):
                continue
            sd = sub["size"].get("d", 0.2)
            target_back = tgt["position"][1] + tgt["size"].get("d", 0.0) / 2.0
            sub["position"][1] = round(target_back + sd / 2.0 + 0.15, 3)
            geom = sub.get("geometry") or {}
            pl = sub.get("placement") or {}
            if geom.get("coordinate_mode") == "center" or pl.get("anchor") == "center":
                sub["position"][2] = round(sub["size"].get("h", 0.0) / 2.0, 3)
            self.record(sub["id"], "position", sub["position"], f"behind:{tgt['id']}")

        # Overhead trusses centered/placed above the referenced stage.
        for r in rels:
            if r.get("type") not in ("centered_on", "above"):
                continue
            sub = self.by_id.get(r.get("subject"))
            tgt = self.by_id.get(r.get("target"))
            if not (sub and tgt and sub.get("position") and tgt.get("position")):
                continue
            if sub.get("category") == "truss" and (sub.get("placement") or {}).get("zone") == "overhead":
                sub["position"][0] = tgt["position"][0]
                sub["position"][1] = tgt["position"][1]
                if r.get("type") == "above":
                    clearance = self.stage.get("height_clearance", 6.0)
                    sub["position"][2] = max(sub["position"][2], round(clearance * 0.80, 3))
                self.record(sub["id"], "position", sub["position"], f"{r.get('type')}_over:{tgt['id']}")

        # Distribute lights mounted on the same truss instead of stacking them.
        groups = {}
        for o in self.objects:
            if o.get("category") != "light":
                continue
            pl = o.get("placement") or {}
            mount = pl.get("mounted_on") or next(iter(relation_targets.get(("mounted_on", o["id"]), [])), None)
            if mount:
                groups.setdefault(mount, []).append(o)
        for mount_id, lights in groups.items():
            mount = self.by_id.get(mount_id)
            if not (mount and mount.get("position")):
                continue
            mw = (mount.get("size") or {}).get("w", max(1.0, len(lights)))
            n = len(lights)
            for i, light in enumerate(lights):
                if n == 1:
                    offset = 0.0
                else:
                    offset = -mw * 0.38 + (mw * 0.76) * i / (n - 1)
                light["position"] = [round(mount["position"][0] + offset, 3),
                                     round(mount["position"][1], 3),
                                     round(mount["position"][2] - 0.25, 3)]
                self.record(light["id"], "position", light["position"], f"mounted_on:{mount_id}:distributed")

    def fill_rotations_source_confidence(self):
        for o in self.objects:
            if o.get("rotation") is None:
                o["rotation"] = [0.0, 0.0, 0.0]
                self.record(o["id"], "rotation", [0.0, 0.0, 0.0], "default_no_rotation")
            if o.get("source") is None:
                o["source"] = DEFAULT_SOURCE
                self.record(o["id"], "source", DEFAULT_SOURCE, "default_source")
            if o.get("confidence") is None:
                conf = CONF_BY_SOURCE.get(o["source"], 0.3)
                o["confidence"] = conf
                self.record(o["id"], "confidence", conf, f"inferred_from_source:{o['source']}")

    def _read_stair_step_count(self, o):
        """Return an explicit visual/design step count when present.

        Step count is a visible design attribute and must not be inferred from
        the final total height after any access-height adjustment.  Look for it
        in multiple schema-compatible locations and common textual cues.
        """
        holders = []
        if isinstance(o, dict):
            holders.append(o)
            for key in ("generation_hints", "attributes"):
                if isinstance(o.get(key), dict):
                    holders.append(o[key])
            geom = o.get("geometry") if isinstance(o.get("geometry"), dict) else {}
            holders.append(geom)
            for key in ("procedural", "detail_params"):
                if isinstance(geom.get(key), dict):
                    holders.append(geom[key])
        for holder in holders:
            for key in ("visual_step_count", "visible_step_count", "design_step_count", "step_count", "steps", "tread_count"):
                if isinstance(holder, dict) and holder.get(key) is not None:
                    try:
                        n = int(round(float(holder.get(key))))
                        if 1 <= n <= 30:
                            return n, f"explicit:{key}"
                    except Exception:
                        pass

        text = self._semantic_text(o) + " " + json.dumps(o.get("source_evidence", []), ensure_ascii=False).lower()
        cn = {"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
        for ch, n in cn.items():
            if any(p in text for p in (f"{ch}级", f"{ch}层", f"{ch}阶", f"{ch}步")):
                return n, "text_cn_count"
        import re
        m = re.search(r"(\d+)\s*(?:steps|step|treads|tread|级|层|阶|步)", text)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 30:
                    return n, "text_numeric_count"
            except Exception:
                pass
        words = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10}
        for word, n in words.items():
            if any(p in text for p in (f"{word} steps", f"{word}-step", f"{word} treads")):
                return n, "text_word_count"
        return None, None

    def infer_stair_visual_step_counts(self):
        """Preserve visual stair/tread counts independently of height.

        A later access-height policy may cap or adjust total height to avoid a
        default overshoot, but it must not reduce the visible number of treads.
        When no explicit count is available, estimate count from stair depth/run
        rather than height, because the height may be intentionally low or capped.
        """
        for o in self.objects:
            if o.get("category") not in ("stairs", "ramp") or not o.get("size"):
                continue
            n, source = self._read_stair_step_count(o)
            if n is None:
                s = o.get("size") or {}
                d = float(s.get("d", 0.0) or 0.0)
                # Generic stage/exhibit access heuristic: visible tread count is
                # primarily a depth/run property, not a height property.  Clamp to
                # avoid hallucinating very large counts from rough VLM dimensions.
                if d > 0:
                    n = int(math.ceil(d / 0.30))
                    n = max(1, min(12, n))
                    source = "depth_based_visual_run_estimate"
            if n is None:
                continue
            geom = o.setdefault("geometry", {})
            geom.setdefault("representation", "procedural")
            proc = geom.setdefault("procedural", {})
            proc.setdefault("shape_family", "straight_stairs")
            # Do not overwrite a stronger explicit count, but make the count
            # available to generators under all common names.
            proc["step_count"] = int(n)
            proc.setdefault("tread_count", int(n))
            gh = o.setdefault("generation_hints", {})
            gh["visual_step_count"] = int(n)
            gh["step_count_source"] = source or "unknown"
            # Preserve the original visual/parsed height before access policies
            # alter size.h.  Downstream can compare, but should keep step count.
            try:
                gh.setdefault("visual_total_height", float((o.get("size") or {}).get("h")))
            except Exception:
                pass
            self.record(o["id"], "geometry.procedural.step_count", int(n), "visual_step_count_preserved:" + str(source))

    def infer_access_connection_constraints(self):
        """Infer safe but non-dogmatic access constraints for stairs/ramps.

        This does NOT force every stair to be flush with the target platform.
        It only prevents unsupported overshoot by default.  If visual/user
        evidence says the stairs are decorative, intentionally taller, lower, or
        not a real access path, the height is preserved and marked.
        """
        stage = self._primary_stage_object()
        if not stage or not (stage.get("position") and stage.get("size")):
            return
        target_top = float(stage["position"][2]) + float((stage.get("size") or {}).get("h", 0.0) or 0.0)

        # Map attached_to relations for access objects.
        attached = {}
        for r in self.scene.get("relations", []) or []:
            if r.get("type") in ("attached_to", "connects_to", "adjacent_to") and r.get("subject") and r.get("target"):
                attached.setdefault(r.get("subject"), r.get("target"))

        for o in self.objects:
            if o.get("category") not in ("stairs", "ramp") or not o.get("size") or not o.get("position"):
                continue
            target_id = (o.get("placement") or {}).get("attached_to") or attached.get(o.get("id"))
            if target_id != stage.get("id"):
                continue

            text = self._semantic_text(o) + " " + json.dumps(o.get("source_evidence", []), ensure_ascii=False).lower()
            if any(k in text for k in ("decorative", "ornamental", "visual only", "装饰", "造型")):
                ctype = "decorative_steps"
            elif any(k in text for k in ("flush", "level with", "same height", "齐平", "同高")):
                ctype = "flush_access"
            elif any(k in text for k in ("lower than", "step up", "not flush", "低于", "跨一步")):
                ctype = "step_up_access"
            else:
                ctype = "unknown_access"

            gh = o.setdefault("generation_hints", {})
            gh["access_connection_type"] = ctype
            gh["target_surface"] = stage.get("id") + ".top_surface"
            gh["height_policy"] = "preserve_visual_count; cap_default_height_only_when_no_explicit_visual_height"

            if ctype == "decorative_steps" or self._allows_visual_overhang(o):
                self.record(o["id"], "generation_hints.access_connection_type", ctype, "access_connection_inferred_preserve_visual")
                continue

            base_z = float(o["position"][2])
            max_h = max(0.05, target_top - base_z)
            current_h = float((o.get("size") or {}).get("h", 0.0) or 0.0)
            if ctype == "flush_access" and current_h < max_h - 0.05:
                # Explicit flush evidence may raise a too-low stair; this is the
                # only case where we actively equalize height.
                o["size"]["h"] = round(max_h, 3)
                self.record(o["id"], "size.h", o["size"]["h"], "flush_access_target_surface_height")
            elif current_h > max_h + 0.02:
                gh.setdefault("target_top_z", round(target_top, 3))
                gh.setdefault("pre_constraint_total_height", round(current_h, 3))
                # If the source explicitly describes a non-flush/decorative/visual
                # height, preserve it.  Otherwise cap only the total height, never
                # the visible tread count (that is stored separately above).
                if ctype in ("step_up_access", "unknown_access") and not any(k in text for k in ("higher than", "taller than", "高于", "超过", "visual height", "按图")):
                    o["size"]["h"] = round(max_h, 3)
                    gh["height_policy_applied"] = "default_no_overshoot_total_height"
                    self.record(o["id"], "size.h", o["size"]["h"], "access_no_overshoot_target_surface_preserve_step_count")
                else:
                    gh["height_policy_applied"] = "preserve_explicit_visual_height"
                    self.record(o["id"], "generation_hints.height_policy_applied", gh["height_policy_applied"], "access_height_preserved")

    def enforce_surface_containment(self):
        """Keep objects resting on a platform inside its support footprint unless visuals say otherwise."""
        stage = self._primary_stage_object()
        prof = self._stage_profile(stage)
        if not (stage and prof):
            return
        stage_id = stage.get("id")
        on_top_subjects = {r.get("subject") for r in self.scene.get("relations", []) or []
                           if r.get("type") == "on_top_of" and r.get("target") == stage_id}

        for o in self.objects:
            if not (o.get("position") and o.get("size")):
                continue
            if o.get("id") not in on_top_subjects and (o.get("placement") or {}).get("target") != stage_id:
                continue
            if self._allows_visual_overhang(o):
                gh = o.setdefault("generation_hints", {})
                gh["surface_containment"] = "visual_overhang_allowed"
                continue
            # Only adjust small surface equipment.  Do not pull walls/curtains/trusses
            # into the platform footprint.
            if o.get("category") not in ("audio", "prop", "furniture", "light", "screen"):
                continue
            if self._is_line_array_audio(o):
                continue

            x, y, z = [float(v) for v in o["position"][:3]]
            osize = o.get("size") or {}
            hw = float(osize.get("w", 0.0) or 0.0) / 2.0
            hd = float(osize.get("d", 0.0) or 0.0) / 2.0
            if prof.get("is_rect_semicircle"):
                if not self._profile_contains_rect_semicircle_bbox(prof, x, y, hw, hd):
                    zone = (o.get("placement") or {}).get("zone", "")
                    nx, ny = self._clamp_to_rect_semicircle_profile(prof, x, y, hw, hd, preferred_zone=zone or "downstage_apron")
                    if abs(nx - x) > 1e-4 or abs(ny - y) > 1e-4:
                        o["position"][0] = nx
                        o["position"][1] = ny
                        gh = o.setdefault("generation_hints", {})
                        gh["surface_containment"] = "projected_inside_support_footprint"
                        self.record(o["id"], "position", o["position"], "surface_containment_rect_semicircle")
            else:
                # Conservative rectangular fallback for other platform types.
                sx, sy = prof["x"], prof["y"]
                xmin = prof["left_x"] + hw + 0.05
                xmax = prof["right_x"] - hw - 0.05
                ymin = prof["front_y"] + hd + 0.05
                ymax = prof["back_y"] - hd - 0.05
                nx = min(max(x, xmin), xmax) if xmin <= xmax else x
                ny = min(max(y, ymin), ymax) if ymin <= ymax else y
                if abs(nx - x) > 1e-4 or abs(ny - y) > 1e-4:
                    o["position"][0] = round(nx, 3)
                    o["position"][1] = round(ny, 3)
                    gh = o.setdefault("generation_hints", {})
                    gh["surface_containment"] = "projected_inside_support_footprint"
                    self.record(o["id"], "position", o["position"], "surface_containment_rectangular_fallback")

    def normalize_multiview_structure(self):
        """Generic repair for VLM multi-view coordinate and layout artifacts.

        MULTIVIEW_STRUCTURE_NORMALIZATION_V2:
        - image/depth coordinates sometimes land in z; ground unsupported
          floor objects and vertical surfaces before relation solving.
        - preserve side/symmetric curtains/walls using generation_hints.layout.
        - recover mounted_on / on_top_of / aligned_with targets from compact
          layout hints when relation sanitization had weak evidence.
        """
        W = float(self.stage.get("width", 12.0) or 12.0)
        D = float(self.stage.get("depth", 8.0) or 8.0)
        H = float(self.stage.get("height_clearance", 6.0) or 6.0)

        # 1) Suspicious z repair. High z is valid for lights/trusses/ceiling
        # hardware, but not for ordinary furniture/platforms/walls from image.
        for o in self.objects:
            pos = o.get("position")
            if not (isinstance(pos, list) and len(pos) >= 3):
                continue
            cat = o.get("category")
            subtype = str(o.get("subtype", "")).lower()
            z = pos[2]
            try:
                zf = float(z)
            except Exception:
                continue
            if cat in ("light", "truss", "rigging", "ceiling_piece"):
                continue
            if cat in ("furniture", "platform", "stairs", "audio", "prop") and zf > 2.0:
                pos[2] = 0.0
                self.record(o.get("id", "?"), "position.z", 0.0, "multiview_z_depth_pollution_repair")
            elif cat in ("wall", "curtain", "scenic_background", "screen") and zf > H * 0.55:
                pos[2] = 0.0
                self.record(o.get("id", "?"), "position.z", 0.0, "multiview_vertical_surface_base_repair")
            elif cat == "decor" and zf > 2.5:
                pos[2] = 0.0
                self.record(o.get("id", "?"), "position.z", 0.0, "multiview_mounted_decor_z_repair")

        # 2) Side/symmetric layout protection for objects carried as compact arrays.
        for o in self.objects:
            gh = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
            layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else {}
            positions = layout.get("positions") if isinstance(layout.get("positions"), list) else []
            if not positions:
                continue
            cat = o.get("category")
            txt = self._semantic_text(o) if hasattr(self, "_semantic_text") else json.dumps(o, ensure_ascii=False).lower()
            if cat in ("curtain", "wall", "scenic_background") or "side" in txt or "symmetric" in str(layout.get("type", "")).lower():
                xs = [float(p[0]) for p in positions if isinstance(p, list) and len(p) >= 2 and isinstance(p[0], (int, float))]
                ys = [float(p[1]) for p in positions if isinstance(p, list) and len(p) >= 2 and isinstance(p[1], (int, float))]
                if xs and max(xs) - min(xs) > W * 0.35:
                    # Keep this compact object centered as an array proxy, but
                    # record its actual lateral span for the generator.
                    o.setdefault("generation_hints", {})["preserve_side_positions"] = positions
                    o["position"] = [round((min(xs)+max(xs))/2.0, 3), round(sum(ys)/len(ys), 3) if ys else D*0.5, 0.0]
                    o.setdefault("placement", {})["side"] = "both"
                    self.record(o["id"], "position", o["position"], "preserve_symmetric_side_layout_proxy")

        # 3) Recover relation targets from layout hints when present.
        for o in self.objects:
            gh = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
            layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else {}
            oid = o.get("id")
            if not oid:
                continue
            if layout.get("attached_to"):
                self._add_relation_once("attached_to", oid, str(layout.get("attached_to")), source="validator", confidence=0.62)
            if layout.get("mounted_on"):
                self._add_relation_once("mounted_on", oid, str(layout.get("mounted_on")), source="validator", confidence=0.62)
            if layout.get("target"):
                self._add_relation_once("aligned_with", oid, str(layout.get("target")), source="validator", confidence=0.55)


    # ---- Information fidelity V3: keep raw semantics before solving geometry ----
    def _object_role(self, o):
        """Classify generic generation role from schema fields, not scene names.

        The role is used only as a hint for downstream passes/generator. It is
        deliberately category/subtype/layout/relation based so it generalizes to
        theatre stages, concerts, fashion runways, conference stages, etc.
        """
        cat = str(o.get("category", "")).lower()
        subtype = str(o.get("subtype", "")).lower()
        gh = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
        layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else {}
        text = self._semantic_text(o) if hasattr(self, "_semantic_text") else json.dumps(o, ensure_ascii=False).lower()
        if gh.get("count") or gh.get("prototype") or layout:
            return "array_proxy"
        if cat in ("platform", "stairs", "ramp", "floor"):
            return "support_surface"
        if cat in ("wall", "curtain", "screen", "scenic_background"):
            return "vertical_surface"
        if cat in ("truss", "rigging", "ceiling_piece", "light") or any(k in text for k in ("overhead", "ceiling", "suspended", "吊", "顶")):
            return "suspended_object"
        if cat in ("decor", "floor_marking"):
            return "mounted_or_surface_detail"
        if cat in ("furniture", "prop", "audio", "set_piece", "column", "arch", "rail"):
            return "ground_supported"
        return "unknown"

    def normalize_information_fidelity(self):
        """Preserve the information that raw VLM output already got right.

        This pass intentionally does not hard-code any particular object.  It
        normalizes relations/materials/layout/prototypes into canonical fields
        and stores raw coordinates as evidence, because raw multi-view positions
        are often low-confidence but still useful for matching platforms/groups.
        """
        # relation aliases should all leave a usable target field.
        normalized = []
        seen = set()
        for r in self.scene.get("relations", []) or []:
            if not isinstance(r, dict):
                continue
            rtype = r.get("type") or r.get("relation") or r.get("relation_type")
            subject = r.get("subject") or r.get("from") or r.get("source_object")
            target = r.get("target") or r.get("object") or r.get("to") or r.get("target_id") or r.get("attached_to") or r.get("mounted_on")
            if isinstance(r.get("relation"), str):
                parsed_type, parsed_target = _parse_relation_text(r.get("relation"))
                rtype = rtype or parsed_type
                target = target or parsed_target
            if rtype not in REL_TYPES:
                rtype = "connected_to" if subject and target else None
            if not (rtype and subject):
                continue
            nr = dict(r)
            nr["type"] = str(rtype)
            nr["subject"] = str(subject)
            if target is not None:
                nr["target"] = str(target)
            # Keep legacy object alias as evidence only if present, but target is canonical.
            key = (nr.get("type"), nr.get("subject"), nr.get("target"))
            if key in seen:
                continue
            seen.add(key)
            normalized.append(nr)
        self.scene["relations"] = normalized

        # Copy layout/prototype/count to generation_hints and preserve raw positions.
        for o in self.objects:
            gh = o.setdefault("generation_hints", {}) if isinstance(o.get("generation_hints"), dict) else {}
            if not isinstance(o.get("generation_hints"), dict):
                o["generation_hints"] = gh
            if o.get("count") is not None and gh.get("count") is None:
                gh["count"] = o.get("count")
            if isinstance(o.get("layout"), dict) and gh.get("layout") is None:
                gh["layout"] = self._copy_json(o.get("layout"))
            if isinstance(o.get("prototype"), dict) and gh.get("prototype") is None:
                gh["prototype"] = self._copy_json(_sanitize_prototype(o.get("prototype")))
            if isinstance(o.get("position"), list) and len(o.get("position")) >= 3 and gh.get("raw_position") is None:
                gh["raw_position"] = self._copy_json(o.get("position"))
            role = self._object_role(o)
            if role != "unknown" and gh.get("role") is None:
                gh["role"] = role
            if role == "array_proxy" and gh.get("must_expand_array") is None:
                gh["must_expand_array"] = True

            # Normalize object material/color repeatedly because later passes may
            # introduce material strings from groups/prototypes.
            _normalize_material_color_fields(o, o)
            proto = gh.get("prototype")
            if isinstance(proto, dict):
                _normalize_material_color_fields(proto, proto)

            # Repair layout positions for downstream consumers without deleting
            # raw evidence.  For ground/vertical/support arrays, z in layout
            # positions is often image-depth; keep it in raw_layout and expose
            # sanitized positions as z=0. Suspended layouts keep their z.
            layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else None
            if layout:
                if gh.get("raw_layout") is None:
                    gh["raw_layout"] = self._copy_json(layout)
                role = gh.get("role") or self._object_role(o)
                if isinstance(layout.get("positions"), list):
                    fixed_positions = []
                    changed = False
                    for p in layout.get("positions"):
                        if isinstance(p, list) and len(p) >= 2:
                            q = [p[0], p[1], p[2] if len(p) >= 3 else 0.0]
                            if role != "suspended_object" and isinstance(q[2], (int, float)) and abs(float(q[2])) > 2.0:
                                q[2] = 0.0
                                changed = True
                            fixed_positions.append(q)
                    if fixed_positions and changed:
                        layout["positions"] = fixed_positions
                if isinstance(layout.get("center"), list) and len(layout.get("center")) >= 3:
                    c = list(layout.get("center"))
                    if role != "suspended_object" and isinstance(c[2], (int, float)) and abs(float(c[2])) > 2.0:
                        c[2] = 0.0
                        layout["center"] = c

        # Ensure group distributions are preserved and get sanitized raw layout evidence.
        for g in self.scene.get("object_groups", []) or []:
            if not isinstance(g, dict):
                continue
            if isinstance(g.get("layout"), dict) and not isinstance(g.get("distribution"), dict):
                g["distribution"] = self._copy_json(g.get("layout"))
            dist = g.get("distribution") if isinstance(g.get("distribution"), dict) else {}
            if dist and not g.get("raw_distribution"):
                g["raw_distribution"] = self._copy_json(dist)
            proto = g.get("prototype") if isinstance(g.get("prototype"), dict) else None
            if proto:
                _normalize_material_color_fields(proto, proto)
        self.refresh_index()

    def _relation_subjects(self):
        return {r.get("subject") for r in self.scene.get("relations", []) or [] if isinstance(r, dict) and r.get("subject")}

    def _group_expanded_ids(self, gid):
        ids = self.expanded_group_map.get(gid)
        if ids:
            return list(ids)
        return [o.get("id") for o in self.objects if o.get("group_id") == gid and o.get("id")]

    def _choose_target_from_group(self, subject, gid):
        """Choose the best expanded object from a group target without scene-specific names."""
        ids = self._group_expanded_ids(gid)
        cands = [self.by_id.get(i) for i in ids if self.by_id.get(i)]
        cands = [c for c in cands if c.get("position") and c.get("size")]
        if not cands:
            return None
        gh = subject.get("generation_hints") if isinstance(subject.get("generation_hints"), dict) else {}
        raw_pos = gh.get("raw_position") if isinstance(gh.get("raw_position"), list) else subject.get("position")
        if isinstance(raw_pos, list) and len(raw_pos) >= 2:
            x, y = float(raw_pos[0]), float(raw_pos[1])
            return min(cands, key=lambda c: (float(c["position"][0]) - x) ** 2 + (float(c["position"][1]) - y) ** 2)
        return cands[0]

    def solve_relation_anchors_generic(self):
        """Generic relation-driven positioning.

        This is intentionally lightweight: it only applies high-confidence
        structural consequences of relations.  It avoids object-name rules, and
        it does not try to make a final aesthetic layout.
        """
        rels = self.scene.get("relations", []) or []
        for r in rels:
            if not isinstance(r, dict):
                continue
            rtype, sid, tid = r.get("type"), r.get("subject"), r.get("target")
            if not (rtype and sid and tid):
                continue
            sub = self.by_id.get(sid)
            if not sub:
                continue
            target = self.by_id.get(tid)
            # If the target is a group id, resolve to the nearest expanded child.
            if target is None:
                target = self._choose_target_from_group(sub, tid)
                if target is not None:
                    r.setdefault("target_resolved", target.get("id"))
                    sub.setdefault("placement", {})["target_surface"] = target.get("id")
                    self._add_relation_once(rtype, sid, target.get("id"), source="validator", confidence=0.7)
            if not target:
                continue
            if rtype == "on_top_of" and sub.get("position") and target.get("position") and target.get("size"):
                top = float(target["position"][2]) + float((target.get("size") or {}).get("h", 0.0) or 0.0)
                if abs(float(sub["position"][2]) - top) > OVERLAP_EPS:
                    sub["position"][2] = round(top, 3)
                    self.record(sid, "position.z", sub["position"][2], f"relation_on_top_of:{target.get('id')}")
            elif rtype in ("mounted_on", "attached_to") and target.get("position") and target.get("size"):
                tcat = target.get("category")
                if tcat in ("wall", "curtain", "screen", "scenic_background"):
                    # Keep subject as a proxy if it is an array; generator can
                    # expand it on the target surface using its own layout.
                    gh = sub.setdefault("generation_hints", {})
                    gh["mount_target"] = target.get("id")
                    if sub.get("category") in ("decor", "screen", "floor_marking") and sub.get("position"):
                        tx, ty, tz = target["position"][:3]
                        ts = target.get("size") or {}
                        sub["position"][1] = round(float(ty) - max(0.02, float(ts.get("d", 0.1) or 0.1) / 2.0) - 0.01, 3)
                        sub["position"][2] = round(float(tz) + float(ts.get("h", 2.5) or 2.5) * 0.58, 3)
                        self.record(sid, "position", sub["position"], f"relation_mounted_on_vertical_surface:{target.get('id')}")
                elif target.get("category") in ("truss", "rigging", "ceiling_piece") and sub.get("position"):
                    gh = sub.setdefault("generation_hints", {})
                    gh["mount_target"] = target.get("id")
                    tx, ty, tz = target["position"][:3]
                    th = float((target.get("size") or {}).get("h", 0.2) or 0.2)
                    sub["position"] = [float(tx), float(ty), round(float(tz) - th / 2.0 - 0.15, 3)]
                    self.record(sid, "position", sub["position"], f"relation_mounted_on_overhead_support:{target.get('id')}")


    # ---- geometry cleanup (runs AFTER fill-in, BEFORE conflict detection) ----
    def _clamp_into_bounds(self, o, W, D, H):
        """Shift a physical object minimally so its footprint sits inside the stage."""
        ax = self.aabb(o)
        dx = dy = 0.0
        if W is not None:
            if ax[0] < -W / 2:
                dx = -W / 2 - ax[0]
            elif ax[1] > W / 2:
                dx = W / 2 - ax[1]
        if D is not None:
            if ax[2] < 0:
                dy = -ax[2]
            elif ax[3] > D:
                dy = D - ax[3]
        if abs(dx) > OVERLAP_EPS or abs(dy) > OVERLAP_EPS:
            o["position"][0] += dx
            o["position"][1] += dy
            self.record(o["id"], "position", o["position"], "clamped_into_stage")
        # vertical: keep bottom on/above floor and top under the ceiling
        h = o["size"]["h"]
        if o["position"][2] < -OVERLAP_EPS:
            o["position"][2] = 0.0
        if H is not None and o["position"][2] + h > H + OVERLAP_EPS:
            o["position"][2] = max(0.0, H - h)

    def cleanup_geometry(self):
        """Fix the systematic placement errors VLMs make: floating/forward objects,
        things not resting on what they sit on, and footprints hanging off the stage.
        Uses the relations the model provided where they are reliable."""
        W = self.stage.get("width")
        D = self.stage.get("depth")
        H = self.stage.get("height_clearance")
        rels = self.scene.get("relations", [])
        on_top = {r["subject"]: r["target"] for r in rels
                  if r.get("type") == "on_top_of" and r.get("subject") and r.get("target")}
        above_subjects = {r["subject"] for r in rels if r.get("type") == "above"}

        # 1) ground-snap only platforms/screens (props like flown speakers keep their height)
        for o in self.objects:
            if o.get("category") not in ("platform", "screen"):
                continue
            oid = o["id"]
            if o.get("parent") or oid in on_top or oid in above_subjects:
                continue
            if o.get("position") and o["position"][2] > OVERLAP_EPS:
                old = round(o["position"][2], 2)
                o["position"][2] = 0.0
                self.record(oid, "position.z", 0.0, f"ground_snap(was_{old})")

        # 2) on_top_of relation: rest the subject on the target's top (keep its x,y)
        for sub_id, tgt_id in on_top.items():
            sub, tgt = self.by_id.get(sub_id), self.by_id.get(tgt_id)
            if not (sub and tgt and sub.get("position") and tgt.get("position") and tgt.get("size")):
                continue
            top = tgt["position"][2] + tgt["size"].get("h", 0.0)
            if abs(sub["position"][2] - top) > OVERLAP_EPS:
                sub["position"][2] = top
                self.record(sub_id, "position.z", top, f"rest_on:{tgt_id}")

        # 3) relative placement: put the subject just past the target's edge.
        #    behind/in_front_of act on Y; left_of/right_of act on X.
        for r in rels:
            t = r.get("type")
            if t not in ("behind", "in_front_of", "left_of", "right_of"):
                continue
            sub, tgt = self.by_id.get(r.get("subject")), self.by_id.get(r.get("target"))
            if not (sub and tgt and sub.get("position") and sub.get("size")
                    and tgt.get("position") and tgt.get("size")):
                continue
            sw = sub["size"].get("w", 0.0)
            sd = sub["size"].get("d", 0.0)
            sub_ax, tgt_ax = self.aabb(sub), self.aabb(tgt)
            if t == "behind" and sub_ax[2] < tgt_ax[3] - OVERLAP_EPS:
                sub["position"][1] = tgt_ax[3] + sd / 2.0
                self.record(r["subject"], "position.y", round(sub["position"][1], 2), f"behind:{r.get('target')}")
            elif t == "in_front_of" and sub_ax[3] > tgt_ax[2] + OVERLAP_EPS:
                sub["position"][1] = tgt_ax[2] - sd / 2.0
                self.record(r["subject"], "position.y", round(sub["position"][1], 2), f"in_front_of:{r.get('target')}")
            elif t == "left_of" and sub_ax[1] > tgt_ax[0] + OVERLAP_EPS:
                sub["position"][0] = tgt_ax[0] - sw / 2.0
                self.record(r["subject"], "position.x", round(sub["position"][0], 2), f"left_of:{r.get('target')}")
            elif t == "right_of" and sub_ax[0] < tgt_ax[1] - OVERLAP_EPS:
                sub["position"][0] = tgt_ax[1] + sw / 2.0
                self.record(r["subject"], "position.x", round(sub["position"][0], 2), f"right_of:{r.get('target')}")

        # 3) Do NOT globally shift the whole scene.  Multi-view VLM coordinates
        # may use negative/positive y inconsistently; a global shift destroys
        # left/right/backdrop/platform relations.  Relation-aware passes should
        # resolve support and mounting instead.

        # 4) clamp only small movable floor objects.  Never clamp structural
        # envelopes (walls/curtains/trusses/platform groups/screens/audience)
        # because that folds the stage into one corner.
        protected = {"wall", "curtain", "scenic_background", "truss", "screen", "platform", "ceiling_piece"}
        relational_subjects = self._relation_subjects() if hasattr(self, "_relation_subjects") else set()
        for o in self.objects:
            if o.get("category") in protected or o.get("category") == "light" or not o.get("position") or not o.get("size"):
                continue
            if o.get("group_id"):
                continue
            # Information-fidelity rule: do not clamp objects whose position is
            # controlled by a relation or an explicit layout/prototype/count.
            # Clamping these collapses multi-view layouts into the stage center.
            gh = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
            if o.get("id") in relational_subjects or gh.get("layout") or gh.get("prototype") or gh.get("count"):
                continue
            self._clamp_into_bounds(o, W, D, H)

    # ---- conflict detection (runs AFTER fill-in, on complete data) ----
    def aabb(self, o):
        """Axis-aligned bounding box [xmin,xmax,ymin,ymax,zmin,zmax] in world space.
        Accounts for yaw (rotation about Z); rx/ry are ignored in v0."""
        px, py, pz = o["position"]
        s = o["size"]
        w, d, h = s["w"], s["d"], s["h"]
        rz = math.radians((o.get("rotation") or [0, 0, 0])[2])
        # yaw-expanded half-extents in the XY plane
        hx = abs(w / 2 * math.cos(rz)) + abs(d / 2 * math.sin(rz))
        hy = abs(w / 2 * math.sin(rz)) + abs(d / 2 * math.cos(rz))
        return [px - hx, px + hx, py - hy, py + hy, pz, pz + h]

    def add_conflict(self, ctype, severity, objects, message):
        self.conflicts.append({"type": ctype, "severity": severity,
                               "objects": objects, "message": message})

    def _resting_pair(self, a, b):
        """True if a rests on b or vice-versa (parent link or on_top_of relation),
        in which case touching is intended, not a conflict."""
        if a.get("parent") == b["id"] or b.get("parent") == a["id"]:
            return True
        for r in self.scene.get("relations", []):
            pair = {r.get("subject"), r.get("target")}
            if r.get("type") == "on_top_of" and pair == {a["id"], b["id"]}:
                return True
            # Stairs and access ramps physically attach to a platform edge. AABB
            # overlap is often a false positive for composite/curved platforms,
            # so treat attached stairs/ramp-platform pairs as intended contact.
            if r.get("type") == "attached_to" and pair == {a["id"], b["id"]}:
                cats = {a.get("category"), b.get("category")}
                if "platform" in cats and ("stairs" in cats or "ramp" in cats):
                    return True
        # Architectural side walls / proscenium columns commonly sit on, intersect
        # or visually embed into the platform/back deck.  This is intended scenic
        # contact, not an actionable collision, as long as one object is platform
        # and the other is an upstage side wall/column/set piece.
        cats = {a.get("category"), b.get("category")}
        if "platform" in cats and cats & {"set_piece", "wall", "column", "scenic_background", "decor"}:
            other = b if a.get("category") == "platform" else a
            txt = (self._semantic_text(other) + " " + str((other.get("placement") or {}).get("zone", ""))).lower()
            if any(k in txt for k in ("proscenium", "side_wall", "side wall", "column", "upstage_side", "flank", "翼", "侧墙", "柱")):
                return True
        # Access stairs, railings and walkway platforms are designed to touch or
        # slightly intersect at connection seams. Treat these as attachments so
        # the conflict list does not drown in expected construction contact.
        text = (self._semantic_text(a) + " " + self._semantic_text(b)).lower()
        if cats & {"stairs", "ramp", "rail"} and cats & {"platform", "rail", "stairs", "ramp"}:
            if any(k in text for k in ("walkway", "railing", "rail", "stairs", "stair", "platform", "走道", "栏杆", "楼梯", "平台")):
                return True
        return False

    def detect_conflicts(self):
        phys = [o for o in self.objects if o.get("category") in PHYSICAL and o.get("size") and o.get("position")]

        # (1) out of bounds: object's box must sit inside the stage envelope
        W = self.stage.get("width")
        D = self.stage.get("depth")
        H = self.stage.get("height_clearance")
        for o in phys:
            xmin, xmax, ymin, ymax, zmin, zmax = self.aabb(o)
            sides = []
            if W is not None and (xmin < -W / 2 - OVERLAP_EPS or xmax > W / 2 + OVERLAP_EPS):
                sides.append("X")
            if D is not None and (ymin < -OVERLAP_EPS or ymax > D + OVERLAP_EPS):
                sides.append("Y (downstage-upstage)")
            if H is not None and zmax > H + OVERLAP_EPS:
                sides.append("Z (height)")
            if sides:
                self.add_conflict(
                    "out_of_bounds", "error", [o["id"]],
                    f"'{o.get('label', o['id'])}' extends past the stage on {', '.join(sides)}."
                )

        # (2) overlap: any two physical boxes that interpenetrate (not merely touch)
        for i in range(len(phys)):
            for j in range(i + 1, len(phys)):
                a, b = phys[i], phys[j]
                if self._resting_pair(a, b):
                    continue
                ax = self.aabb(a)
                bx = self.aabb(b)
                ox = min(ax[1], bx[1]) - max(ax[0], bx[0])
                oy = min(ax[3], bx[3]) - max(ax[2], bx[2])
                oz = min(ax[5], bx[5]) - max(ax[4], bx[4])
                if ox > OVERLAP_EPS and oy > OVERLAP_EPS and oz > OVERLAP_EPS:
                    self.add_conflict(
                        "overlap", "warning", [a["id"], b["id"]],
                        f"'{a.get('label', a['id'])}' and '{b.get('label', b['id'])}' "
                        f"overlap by ~{min(ox, oy, oz):.2f} m. Intended, or should one move?"
                    )

    def write_validation(self):
        existing = self.scene.get("validation", {})
        existing["status"] = "has_conflicts" if self.conflicts else "validated"
        existing["conflicts"] = self.conflicts
        existing["filled"] = self.filled
        self.scene["validation"] = existing

    def run(self):
        from decision_state import initialize, snapshot, reconcile
        initialize(self.scene)
        original_decisions = snapshot(self.scene)
        # Previous conflict results belong to the previous validation run.
        self.scene.setdefault("validation", {})["conflicts"] = []
        self.recover_missing_main_stage()
        self.expand_object_groups()
        self.normalize_placement_relations()
        self.refresh_index()
        self.normalize_information_fidelity()
        self.fill_coordinate_system()
        self.fill_sizes()
        self.normalize_multiview_structure()
        self.normalize_information_fidelity()
        self.recover_layout_semantics()
        self.fill_positions()        # after sizes (needs parent height)
        self.normalize_multiview_structure()
        self.normalize_information_fidelity()
        self.solve_relation_anchors_generic()
        self.fill_rotations_source_confidence()
        self.apply_placement_semantics()
        self.solve_relation_anchors_generic()
        self.distribute_monitor_wedges()
        self.recover_composite_stage_layout_anchors()
        self.infer_stair_visual_step_counts()
        self.infer_access_connection_constraints()
        self.enforce_surface_containment()
        self.cleanup_geometry()      # ground/stack/clamp using relations, before checks
        reconcile(self.scene, original_decisions, self.filled)
        self.conflicts.extend(self.scene.get("validation", {}).get("conflicts", []))
        self.detect_conflicts()      # checks use preserved adopted geometry
        self.write_validation()
        return self.scene


# --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fill in a raw InkVox scene.")
    ap.add_argument("scene", help="raw scene JSON to complete")
    ap.add_argument("-o", "--out", default=None, help="output path (default: <name>.filled.json)")
    args = ap.parse_args()

    src = Path(args.scene)
    out = Path(args.out) if args.out else src.with_name(src.stem + ".filled.json")

    data = json.loads(src.read_text(encoding="utf-8"))
    data = sanitize_scene_json(data)
    if isinstance(data.get("scene"), dict):
        data["scene"]["_schema_version"] = data.get("schema_version")
    v = Validator(data["scene"])
    v.run()
    data["scene"].pop("_schema_version", None)
    data = sanitize_scene_json(data)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    from decision_state import handoff_markdown
    out.with_suffix(".handoff.md").write_text(handoff_markdown(data), encoding="utf-8")
    print(f"Validator version: {INKVOX_VALIDATOR_VERSION}")
    print(f"Filled {len(v.filled)} field(s). Wrote -> {out.name}\n")
    for f in v.filled:
        val = f["value"]
        val = json.dumps(val) if isinstance(val, (list, dict)) else val
        print(f"  {f['object']:<14} {f['field']:<16} = {str(val):<22} ({f['reason']})")

    print(f"\nStatus: {data['scene']['validation']['status']}  |  {len(v.conflicts)} conflict(s)")
    for c in v.conflicts:
        print(f"  [{c['severity']}] {c['type']}: {c['message']}")


if __name__ == "__main__":
    main()
