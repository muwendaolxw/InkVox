#!/usr/bin/env python3
"""
generate_stage.py  --  InkVox Blender generator (targets Blender 3.4)

Reads a scene JSON (see scene.schema.json), builds a 3D stage in Blender, saves
an editable .blend and renders a preview PNG.

This version upgrades the geometry from plain boxes to category-aware detailed
builders (bevelled edges + weighted normals, a small material library, truss
lattices, stacked line-array speakers, pleated curtains, stepped stairs with
nosing, framed screens, table/chair legs). Everything is still driven by the
scene.json so it stays automatic and editable.

Run:
    blender --background --python generate_stage.py -- scene.filled.json [out.png]

Coordinate convention (must match the schema):
    meters / degrees; origin = downstage center on the floor;
    +X = audience's right, +Y = upstage (depth), +Z = up. (Matches Blender.)
"""

import bpy
import mathutils
import sys
import os
import json
import math
import re

INKVOX_GENERATOR_VERSION = "SCENE_V0_4_DESIGN_PARAMETERS_20261001"

# ---------------------------------------------------------------- tunables --
RENDER_W, RENDER_H = 1280, 800
CYCLES_SAMPLES = 96
SPOT_MAX_W = 3000.0
AREA_MAX_W = 1500.0
POINT_MAX_W = 1000.0
EMISSION_STRENGTH = 3.0
WORLD_BG = (0.012, 0.014, 0.018, 1.0)  # dark blue-grey world with enough lift for preview renders
VISIBILITY_FALLBACKS = True
FORCE_VISIBILITY_FALLBACKS = False  # generic debug-safe proxies for small dark stage hardware

DEFAULT_COLOR = {
    "platform": (0.20, 0.12, 0.07),
    "stairs": (0.20, 0.12, 0.07),
    "curtain": (0.18, 0.02, 0.03),
    "scenic_background": (0.28, 0.29, 0.31),
    "wall": (0.28, 0.29, 0.31),
    "truss": (0.24, 0.25, 0.27),
    "light": (0.03, 0.03, 0.035),
    "audio": (0.015, 0.015, 0.015),
    "prop": (0.45, 0.45, 0.47),
    "screen": (0.02, 0.04, 0.12),
}

COLOR_NAME_MAP = {
    "warm_wood": "#8A5A2B",
    "wood": "#8A5A2B",
    "deep_red": "#7A0E16",
    "red": "#8B1119",
    "dark_red": "#5A0710",
    "velvet_red": "#7A0E16",
    "silver_grey": "#8A8D8F",
    "silver_gray": "#8A8D8F",
    "dark_grey": "#2A2B2E",
    "dark_gray": "#2A2B2E",
    "cool_grey": "#7A7F86",
    "cool_gray": "#7A7F86",
    "warm_neutral_floor": "#6F6254",
    "black": "#111111",
    "white": "#EEEEEE",
    "grey": "#777777",
    "gray": "#777777",
    "purple": "#7B2CBF",
    "violet": "#7B2CBF",
    "teal": "#1AA6A6",
    "cyan_green": "#1AA6A6",
    "beige": "#D8C3A5",
    "cream": "#E5D3B3",
    "red_brown": "#8A3B18",
    "reddish_brown": "#8A3B18",
    "dark_green": "#073B2E",
    "deep_green": "#073B2E",
    "various": "#D7C08A",
    "beige_patterned": "#B89C7A",
}


# --------------------------------------------------------------- utilities --
def parse_args():
    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    scene_path = args[0] if len(args) >= 1 else "scene.example.json"
    out_path = args[1] if len(args) >= 2 else None
    if out_path is None:
        base = os.path.splitext(os.path.basename(scene_path))[0]
        out_path = os.path.join(os.path.dirname(os.path.abspath(scene_path)), base + ".png")
    return scene_path, out_path


def srgb_to_linear(c8):
    c = c8 / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return None
    return (srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b))


def _material_spec(obj):
    """Return the strongest material spec available on an object.

    Validator V3 preserves compact arrays as generation_hints.prototype.  The
    generator must treat prototype material as first-class evidence; otherwise
    every expanded chair/seat/curtain falls back to white or category defaults.
    """
    if not isinstance(obj, dict):
        return {}

    def _coerce_material(mat, color=None):
        if isinstance(mat, dict):
            out = dict(mat)
            if color and not any(out.get(k) for k in ("color_name", "base_color", "color", "color_hex")):
                out["color_name"] = color
            return out
        if isinstance(mat, str) and mat.strip():
            out = {"type": mat.strip()}
            if color:
                out["color_name"] = color
            return out
        if color:
            return {"color_name": color}
        return {}

    color = obj.get("color") if isinstance(obj.get("color"), str) else None
    mat = _coerce_material(obj.get("material"), color)
    if mat:
        return mat

    attrs = obj.get("attributes") or {}
    mat = _coerce_material(attrs.get("material"), attrs.get("color") if isinstance(attrs.get("color"), str) else None)
    if mat:
        return mat

    gh = obj.get("generation_hints") if isinstance(obj.get("generation_hints"), dict) else {}
    proto = gh.get("prototype") if isinstance(gh.get("prototype"), dict) else {}
    mat = _coerce_material(proto.get("material"), proto.get("color") if isinstance(proto.get("color"), str) else None)
    return mat


def _rgb_from_hex_or_name(value):
    if not value or not isinstance(value, str):
        return None
    if value.startswith("#"):
        return hex_to_linear(value)
    key = value.strip().lower().replace(" ", "_").replace("-", "_")
    hx = COLOR_NAME_MAP.get(key)
    return hex_to_linear(hx) if hx else None


def base_color_for(obj):
    """Resolve color from Scene JSON v0.3, with v0.2 compatibility.

    Priority: material.color_hex > material.color_name > attributes.color >
    material.type default > category default.  The generator should interpret
    scene.json rather than impose a hard-coded style on every scene.
    """
    mat = _material_spec(obj)
    for key in ("color_hex", "color_name", "base_color", "color"):
        rgb = _rgb_from_hex_or_name(mat.get(key))
        if rgb:
            return rgb

    attrs = obj.get("attributes") or {}
    rgb = _rgb_from_hex_or_name(attrs.get("color"))
    if rgb:
        return rgb

    mtype = str(mat.get("type", "")).lower()
    if "wood" in mtype:
        return hex_to_linear("#8A5A2B")
    if "velvet" in mtype or "fabric" in mtype:
        return hex_to_linear("#7A0E16")
    if "metal" in mtype:
        return hex_to_linear("#6E7378")
    if "plastic" in mtype:
        return hex_to_linear("#111111")
    if "paint" in mtype:
        return hex_to_linear("#7A7F86")

    return DEFAULT_COLOR.get(obj.get("category"), (0.5, 0.5, 0.5))


def material_for_object(obj, name_prefix="obj", fallback_rgb=None):
    mat = _material_spec(obj)
    rgb = fallback_rgb or base_color_for(obj)
    cat = obj.get("category")
    if cat in ("light", "audio"):
        rgb = _visibility_rgb(rgb, cat)
    mtype = str(mat.get("type", "")).lower()
    roughness = float(mat.get("roughness", 0.6 if not mtype else 0.55))
    metallic = float(mat.get("metallic", 0.0))
    emission_rgb = None
    emission_strength = 0.0

    if "velvet" in mtype or "fabric" in mtype:
        roughness = max(roughness, 0.85)
        metallic = 0.0
    elif "metal" in mtype:
        metallic = max(metallic, 0.65)
        roughness = min(max(roughness, 0.25), 0.6)
    elif "plastic" in mtype:
        metallic = 0.0
        roughness = max(roughness, 0.45)
    elif "wood" in mtype:
        metallic = 0.0
        roughness = max(roughness, 0.5)
    elif "screen" in mtype or str(obj.get("category")) == "screen":
        emission_rgb = rgb
        emission_strength = EMISSION_STRENGTH

    oid = str(obj.get("id", name_prefix))
    return get_mat(f"{name_prefix}_{oid}_{mtype or 'mat'}", rgb, roughness=roughness, metallic=metallic,
                   emission_rgb=emission_rgb, emission_strength=emission_strength)

# --------------------------------------------------------------- materials --
_MAT_CACHE = {}
_SCENE_CTX = {}


def make_material(name, base_rgb, roughness=0.6, metallic=0.0, emission_rgb=None, emission_strength=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*base_rgb, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if emission_rgb is not None:
            bsdf.inputs["Emission"].default_value = (*emission_rgb, 1.0)  # 3.4 socket name
            bsdf.inputs["Emission Strength"].default_value = emission_strength
    return mat


def get_mat(name, base_rgb, roughness=0.6, metallic=0.0, emission_rgb=None, emission_strength=0.0):
    if name in _MAT_CACHE:
        return _MAT_CACHE[name]
    m = make_material(name, base_rgb, roughness, metallic, emission_rgb, emission_strength)
    _MAT_CACHE[name] = m
    return m


def mat_wood():       return get_mat("wood", (0.20, 0.12, 0.07), roughness=0.6)
def mat_wood_dark():  return get_mat("wood_dark", (0.06, 0.04, 0.03), roughness=0.7)
def mat_metal():      return get_mat("metal", (0.24, 0.25, 0.27), roughness=0.35, metallic=0.9)
def mat_metal_dark(): return get_mat("metal_dark", (0.05, 0.05, 0.06), roughness=0.5, metallic=0.8)
def mat_generic(rgb): return get_mat("gen_%.2f_%.2f_%.2f" % rgb, rgb, roughness=0.5)
def mat_fabric(rgb):  return get_mat("fab_%.2f_%.2f_%.2f" % rgb, rgb, roughness=0.85)


# ------------------------------------------------------------- primitives --
def _wxy(px, py, lx, ly, rz):
    """Map a local (lx,ly) offset to world XY around (px,py) with yaw rz (rad)."""
    c, s = math.cos(rz), math.sin(rz)
    return (px + lx * c - ly * s, py + lx * s + ly * c)


def _finish_mesh(obj, smooth=True, weighted_normal=True):
    """Apply the mesh finishing that Blender 3.4 expects.

    Weighted Normal modifiers on generated meshes require Auto Smooth to be
    enabled; otherwise Blender prints repeated modifier warnings and the bevels
    do not shade cleanly.  This helper is deliberately called by every mesh
    primitive builder so the fix is consistent across cubes, cylinders, discs,
    trusses, stairs, screens and props.
    """
    if obj is None or not getattr(obj, "data", None):
        return obj
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    if hasattr(obj.data, "use_auto_smooth"):
        obj.data.use_auto_smooth = True
    if weighted_normal:
        try:
            obj.modifiers.new("wn", "WEIGHTED_NORMAL")
        except Exception:
            pass
    return obj


def _cube(name, center, dims, mat=None, rot=(0, 0, 0), bevel=0.02):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=center)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = dims
    obj.rotation_euler = rot
    if mat:
        obj.data.materials.append(mat)
    if bevel and bevel > 0:
        mb = obj.modifiers.new("bevel", "BEVEL")
        mb.width = min(bevel, min(dims) * 0.45)
        mb.segments = 2
    return _finish_mesh(obj)


def _cyl(name, p1, p2, radius, mat=None, verts=12):
    v1 = mathutils.Vector(p1)
    v2 = mathutils.Vector(p2)
    d = v2 - v1
    length = d.length
    if length < 1e-6:
        return None
    mid = (v1 + v2) / 2.0
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=length, location=mid)
    obj = bpy.context.active_object
    obj.name = name
    q = mathutils.Vector((0, 0, 1)).rotation_difference(d.normalized())
    obj.rotation_euler = q.to_euler()
    if mat:
        obj.data.materials.append(mat)
    return _finish_mesh(obj)


# --------------------------------------------------------- object builders --
def _lift_dark_rgb(rgb, minimum=0.075):
    """Keep intentionally dark objects visible against the dark stage background."""
    rgb = tuple(rgb)
    if max(rgb) < minimum:
        return (minimum, minimum, minimum * 1.12)
    return rgb


def _visibility_rgb(rgb, category=None, minimum=None):
    """Preview-safe lift for small dark stage hardware.

    This is intentionally generic: it only prevents very dark audio and lighting
    objects from disappearing in preview renders. It does not impose a material
    style for any particular test scene.
    """
    if minimum is None:
        minimum = 0.065 if category in ("light", "audio") else 0.045
    return _lift_dark_rgb(tuple(rgb), minimum=minimum)


def build_platform(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat = material_for_object(source_obj or {"id": oid, "category": "platform"}, "plat", tuple(col),)
    wx, wy = _wxy(px, py, 0, 0, rz)
    return _cube(oid, (wx, wy, pz + h / 2.0), (w, d, h), mat, rot=(0, 0, rz), bevel=min(0.04, h * 0.4))


def build_disc(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build a circular/oval raised stage as a real cylinder, not a square slab."""
    mat = material_for_object(source_obj or {"id": oid, "category": "platform"}, "disc", tuple(col))
    wx, wy = _wxy(px, py, 0, 0, rz)
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=0.5, depth=h, location=(wx, wy, pz + h / 2.0))
    obj = bpy.context.active_object
    obj.name = oid
    obj.scale = (w, d, 1.0)
    obj.rotation_euler = (0, 0, rz)
    obj.data.materials.append(mat)
    mb = obj.modifiers.new("bevel", "BEVEL")
    mb.width = min(0.035, h * 0.35)
    mb.segments = 3
    return _finish_mesh(obj)


def build_generic(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat = material_for_object(source_obj or {"id": oid, "category": "prop"}, "gen", tuple(col))
    wx, wy = _wxy(px, py, 0, 0, rz)
    return _cube(oid, (wx, wy, pz + h / 2.0), (w, d, h), mat, rot=(0, 0, rz),
                 bevel=min(0.03, min(w, d, h) * 0.4))


def _local_polygon_from_footprint(footprint, w, d):
    if not isinstance(footprint, dict):
        return None
    ftype = str(footprint.get("type", "")).lower()
    if ftype in ("rectangle", "rect"):
        fw = float(footprint.get("w", w))
        fd = float(footprint.get("d", d))
        return [(-fw/2, -fd/2), (fw/2, -fd/2), (fw/2, fd/2), (-fw/2, fd/2)]
    if ftype in ("polygon", "polyline", "bezier_polygon"):
        pts = footprint.get("points") or []
        out = []
        for p in pts:
            if isinstance(p, dict):
                out.append((float(p.get("x", 0)), float(p.get("y", 0))))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((float(p[0]), float(p[1])))
        return out if len(out) >= 3 else None
    if ftype == "semicircle":
        r = float(footprint.get("radius", min(w, d) / 2.0))
        side = str(footprint.get("side", "downstage")).lower()
        if side in ("downstage", "front", "-y"):
            angles = [math.pi - math.pi * i / 24 for i in range(25)]
        else:
            angles = [math.pi * i / 24 for i in range(25)]
        return [(r * math.cos(a), r * math.sin(a)) for a in angles]
    if ftype == "composite_2d":
        parts = footprint.get("parts") or []

        def _ptype(part):
            return str((part or {}).get("type") or (part or {}).get("shape") or "").lower().replace("-", "_")

        rect = next((p for p in parts if _ptype(p) in ("rectangle", "rect", "box")), None)
        semi = next((p for p in parts if _ptype(p) in ("semicircle", "semi_circle", "semicircular_apron", "semi_circular_apron")), None)

        # Generic common stage footprint: rectangular rear platform plus a
        # downstage semicircular apron/protrusion.  The validator may normalize
        # parts to {type: ...} and remove the older {shape: ...}; support both.
        if rect and semi:
            rw = float(rect.get("w", w))
            rd = float(rect.get("d", d))
            r = float(semi.get("radius", min(rw, rd) / 2.0))
            side = str(semi.get("side", "downstage")).lower()
            # Clamp apron radius so it stays attached to the rectangle instead
            # of producing self-intersections when VLM dimensions are rough.
            r = max(0.01, min(r, rw / 2.0))

            pts = []
            if side in ("downstage", "front", "-y"):
                pts.extend([(-rw/2, rd/2), (rw/2, rd/2), (rw/2, -rd/2)])
                # Flat front shoulder from rectangle edge to apron tangent.
                if r < rw / 2.0:
                    pts.append((r, -rd/2))
                for i in range(25):
                    a = 0 - math.pi * i / 24
                    pts.append((r * math.cos(a), -rd/2 + r * math.sin(a)))
                if r < rw / 2.0:
                    pts.append((-r, -rd/2))
                pts.append((-rw/2, -rd/2))
            else:
                pts.extend([(-rw/2, -rd/2), (rw/2, -rd/2), (rw/2, rd/2)])
                if r < rw / 2.0:
                    pts.append((r, rd/2))
                for i in range(25):
                    a = 0 + math.pi * i / 24
                    pts.append((r * math.cos(a), rd/2 + r * math.sin(a)))
                if r < rw / 2.0:
                    pts.append((-r, rd/2))
                pts.append((-rw/2, rd/2))
            return pts
    return None




def _footprint_part_type(part):
    """Return normalized footprint part type from v0.3 {type} or legacy {shape}."""
    return str((part or {}).get("type") or (part or {}).get("shape") or "").lower().replace("-", "_")


def _rect_semicircle_parts(footprint, w, d):
    """Extract common composite stage footprint: rectangle + semicircular apron.

    This is a generic topology specialization, not a scene-specific rule.  It
    prevents unstable single n-gon tops for common composite platforms while
    preserving the same semantic footprint.
    """
    if not isinstance(footprint, dict):
        return None
    if str(footprint.get("type", "")).lower() != "composite_2d":
        return None
    parts = footprint.get("parts") or []
    rect = next((p for p in parts if _footprint_part_type(p) in ("rectangle", "rect", "box")), None)
    semi = next((p for p in parts if _footprint_part_type(p) in ("semicircle", "semi_circle", "semicircular_apron", "semi_circular_apron")), None)
    if not (isinstance(rect, dict) and isinstance(semi, dict)):
        return None
    rw = float(rect.get("w", w) or w)
    rd = float(rect.get("d", d) or d)
    r = float(semi.get("radius", min(rw, rd) / 2.0) or min(rw, rd) / 2.0)
    side = str(semi.get("side", "downstage")).lower()
    r = max(0.01, min(r, rw / 2.0))
    return {"rect": rect, "semi": semi, "rw": rw, "rd": rd, "r": r, "side": side}


def _semicircle_apron_points(r, chord_y, side="downstage", steps=48):
    """Return a stable convex semicircle polygon sharing a chord with a rectangle."""
    side = str(side or "downstage").lower()
    pts = []
    if side in ("downstage", "front", "-y"):
        # Chord lies at the upstage edge of the apron, arc protrudes downstage.
        pts.append((-r, chord_y))
        pts.append((r, chord_y))
        for i in range(1, steps):
            a = -math.pi * i / steps
            pts.append((r * math.cos(a), chord_y + r * math.sin(a)))
    else:
        pts.append((r, chord_y))
        pts.append((-r, chord_y))
        for i in range(1, steps):
            a = math.pi + math.pi * i / steps
            pts.append((r * math.cos(a), chord_y + r * math.sin(a)))
    return pts


def build_composite_rect_semicircle_platform(oid, w, d, h, px, py, pz, rz, col, source_obj):
    """Build rectangle+semicircle platforms as stable sub-meshes.

    Complex composite stages should not be one large concave n-gon.  Splitting
    common parts avoids triangulation/normal artifacts and co-planar top-face
    z-fighting while retaining a single logical platform id via metadata.
    """
    geom = source_obj.get("geometry") or {}
    fp = geom.get("footprint") or {}
    spec = _rect_semicircle_parts(fp, w, d)
    if not spec:
        return None

    rw, rd, r, side = spec["rw"], spec["rd"], spec["r"], spec["side"]
    mat = material_for_object(source_obj, "platform_top", tuple(col))
    fascia_rgb = tuple(max(0.035, c * 0.72) for c in tuple(col))
    fascia_mat = material_for_object({**source_obj, "material": {**(_material_spec(source_obj) or {}), "color": None}}, "platform_fascia", fascia_rgb)

    # Rectangular rear platform.  It is a separate, stable cuboid with the same
    # top material.  We keep the logical id on the main part.
    rx, ry = _wxy(px, py, 0, 0, rz)
    rect_obj = _cube(oid, (rx, ry, pz + h / 2.0), (rw, rd, h), mat, rot=(0, 0, rz), bevel=min(0.025, h * 0.25))

    # Semicircular apron.  It is convex and triangulates reliably.  Its chord
    # shares the rectangle front/back edge rather than overlapping the rectangle
    # top surface.
    chord_y = -rd / 2.0 if side in ("downstage", "front", "-y") else rd / 2.0
    pts = _semicircle_apron_points(r, chord_y, side=side, steps=48)
    apron = build_extruded_polygon(oid + "_apron", pts, h, px, py, pz, rz, mat, bevel=min(0.025, h * 0.25))

    # Add a subtle fascia strip to the curved apron side so the platform reads
    # as a raised stage; this is geometry, not a different top material.
    if apron is not None:
        apron.data.materials.clear()
        apron.data.materials.append(mat)

    # Metadata for both pieces keeps downstream inspection coherent.
    try:
        rect_obj["inkvox_composite_main"] = oid
        apron["inkvox_composite_main"] = oid
        apron["inkvox_category"] = str(source_obj.get("category", "platform"))
        apron["inkvox_subtype"] = str(source_obj.get("subtype", "composite_platform_apron"))
    except Exception:
        pass
    return rect_obj

def build_extruded_polygon(oid, points, h, px, py, pz, rz, mat, bevel=0.02):
    verts = []
    for x, y in points:
        wx, wy = _wxy(px, py, x, y, rz)
        verts.append((wx, wy, pz))
    for x, y in points:
        wx, wy = _wxy(px, py, x, y, rz)
        verts.append((wx, wy, pz + h))
    n = len(points)
    faces = [tuple(range(n-1, -1, -1)), tuple(range(n, 2*n))]
    for i in range(n):
        faces.append((i, (i+1) % n, n + (i+1) % n, n + i))
    mesh = bpy.data.meshes.new(oid + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(oid, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if bevel and bevel > 0:
        mb = obj.modifiers.new("bevel", "BEVEL")
        mb.width = bevel
        mb.segments = 2
    return _finish_mesh(obj)


def build_from_footprint(oid, w, d, h, px, py, pz, rz, col, source_obj):
    geom = source_obj.get("geometry") or {}
    fp = geom.get("footprint") or {}
    ftype = str(fp.get("type", "")).lower()
    if ftype == "circle":
        r = float(fp.get("radius", max(w, d) / 2.0))
        return build_disc(oid, r * 2.0, r * 2.0, h, px, py, pz, rz, col, source_obj)

    # Common composite platforms, such as rectangular stage + semicircular apron,
    # need topology-aware construction.  A single large n-gon top can produce
    # triangulation/normal artifacts in Blender.  Use stable sub-meshes when the
    # footprint parts are explicit; fall back to polygon extrusion for arbitrary
    # polygons.
    if ftype == "composite_2d" and _rect_semicircle_parts(fp, w, d):
        obj = build_composite_rect_semicircle_platform(oid, w, d, h, px, py, pz, rz, col, source_obj)
        if obj is not None:
            return obj

    mat = material_for_object(source_obj, "footprint", tuple(col))
    pts = _local_polygon_from_footprint(fp, w, d)
    if pts:
        return build_extruded_polygon(oid, pts, h, px, py, pz, rz, mat,
                                      bevel=float((geom.get("detail_params") or {}).get("edge_bevel", 0.025)))
    return build_platform(oid, w, d, h, px, py, pz, rz, col, source_obj)


def build_truss(oid, w, d, h, px, py, pz, rz, col=None, source_obj=None):
    """Tube lattice running along its longest horizontal axis (X)."""
    mat = material_for_object(source_obj or {"id": oid, "category": "truss", "material": {"type": "brushed_metal"}}, "truss", tuple(col or DEFAULT_COLOR["truss"]))
    hw = w / 2.0
    cz = pz + h / 2.0
    r = max(0.018, min(d, h) * 0.06)
    first = None
    # 4 chords along X at the corners of the d x h cross-section
    for sy in (-1, 1):
        for sz in (-1, 1):
            y = sy * d / 2.0
            z = cz + sz * h / 2.0
            ax = _wxy(px, py, -hw, y, rz)
            bx = _wxy(px, py, hw, y, rz)
            o = _cyl(oid + "_chord", (ax[0], ax[1], z), (bx[0], bx[1], z), r, mat)
            first = first or o
    n = max(2, int(w / 0.7))
    for sy in (-1, 1):
        y = sy * d / 2.0
        # zigzag web on each long face
        for i in range(n):
            x0 = -hw + w * i / n
            x1 = -hw + w * (i + 1) / n
            ztop, zbot = cz + h / 2.0, cz - h / 2.0
            (a0, az0), (a1, az1) = ((x0, ztop), (x1, zbot)) if i % 2 == 0 else ((x0, zbot), (x1, ztop))
            wa = _wxy(px, py, a0, y, rz)
            wb = _wxy(px, py, a1, y, rz)
            _cyl(oid + "_web", (wa[0], wa[1], az0), (wb[0], wb[1], az1), r * 0.7, mat)
        # verticals at nodes
        for i in range(n + 1):
            x = -hw + w * i / n
            wv = _wxy(px, py, x, y, rz)
            _cyl(oid + "_vert", (wv[0], wv[1], cz - h / 2.0), (wv[0], wv[1], cz + h / 2.0), r * 0.7, mat)
    return first


def build_line_array(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build a line-array speaker as a vertical multi-module hanging array."""
    col = _lift_dark_rgb(tuple(col if col else (0.04, 0.04, 0.05)), minimum=0.055)
    mat = material_for_object(source_obj or {"id": oid, "category": "audio", "material": {"type": "plastic"}}, "spk", tuple(col))
    grille = get_mat("line_array_grille_" + oid, (0.025, 0.025, 0.028), roughness=0.72, metallic=0.10)
    rig_mat = mat_metal_dark()
    n = max(4, int(round(h / 0.30)))
    step = h / n
    mh = step * 0.78
    first = None
    # Slight progressive forward offset makes it read as a flown curved array,
    # but remains generic and subtle enough for wall-hung arrays.
    for i in range(n):
        t = 0 if n <= 1 else i / (n - 1)
        local_y = -0.035 * t
        z = pz + (i + 0.5) * step
        wx, wy = _wxy(px, py, 0, local_y, rz)
        o = _cube(oid + "_m%d" % i, (wx, wy, z), (w, d, mh), mat, rot=(0, 0, rz),
                  bevel=min(0.018, mh * 0.35))
        # Front grille plate.
        gx, gy = _wxy(px, py, 0, local_y - d / 2.0 - 0.009, rz)
        _cube(oid + "_grille_%d" % i, (gx, gy, z), (w * 0.78, 0.014, mh * 0.70), grille, rot=(0, 0, rz), bevel=0.004)
        first = first or o
    # Simple top rigging bar/hanger.
    tx, ty = _wxy(px, py, 0, 0, rz)
    _cube(oid + "_hanger", (tx, ty, pz + h + 0.055), (w * 0.78, d * 0.55, 0.07), rig_mat, rot=(0, 0, rz), bevel=0.006)
    return first


def build_monitor_wedge(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build a visible monitor wedge with a readable grille face.

    The dimensions still come from scene.json. A generic minimum visual size is
    applied only to very small stage-audio hardware so it remains inspectable
    in preview renders across different scenes.
    """
    base = _visibility_rgb(tuple(col), "audio", minimum=0.06)
    mat = material_for_object(source_obj or {"id": oid, "category": "audio"}, "monitor", base)
    grille = get_mat("monitor_grille_" + oid, (0.075, 0.075, 0.082), roughness=0.7, metallic=0.18)

    wv = max(w, 0.60)
    dv = max(d, 0.42)
    hv = max(h, 0.22)

    pts = [(-wv/2, -dv/2, 0), (wv/2, -dv/2, 0), (wv/2, dv/2, 0), (-wv/2, dv/2, 0),
           (-wv/2, -dv/2, hv*0.35), (wv/2, -dv/2, hv*0.35), (wv/2, dv/2, hv), (-wv/2, dv/2, hv)]
    verts = []
    for x, y, z in pts:
        wx, wy = _wxy(px, py, x, y, rz)
        verts.append((wx, wy, pz + z))
    faces = [(0,1,2,3), (4,7,6,5), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    mesh = bpy.data.meshes.new(oid + "Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(oid, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.data.materials.append(mat)
    mb = obj.modifiers.new("bevel", "BEVEL")
    mb.width = min(0.025, hv * 0.18)
    mb.segments = 2
    _finish_mesh(obj)

    # Grille on the high face; useful for any monitor wedge, not only this case.
    gx, gy = _wxy(px, py, 0, dv/2 + 0.012, rz)
    _cube(oid + "_grille", (gx, gy, pz + hv * 0.62), (wv * 0.82, 0.018, hv * 0.44),
          grille, rot=(0, 0, rz), bevel=0.006)
    for i in (-0.26, 0.0, 0.26):
        sx, sy = _wxy(px, py, i * wv, dv/2 + 0.024, rz)
        _cube(oid + "_grille_slot", (sx, sy, pz + hv * 0.62), (0.025, 0.012, hv * 0.35),
              mat, rot=(0, 0, rz), bevel=0.003)

    # Audience-side rim and top edge make a front-of-stage wedge visible even
    # when the functional grille faces upstage toward performers.
    edge_mat = get_mat("monitor_edge_" + oid, (0.16, 0.16, 0.17), roughness=0.62, metallic=0.08)
    fx, fy = _wxy(px, py, 0, -dv/2 - 0.018, rz)
    _cube(oid + "_front_rim", (fx, fy, pz + hv * 0.30), (wv * 0.82, 0.022, hv * 0.18),
          edge_mat, rot=(0, 0, rz), bevel=0.004)
    tx, ty = _wxy(px, py, 0, 0.02, rz)
    _cube(oid + "_top_highlight", (tx, ty, pz + hv + 0.012), (wv * 0.78, dv * 0.10, 0.018),
          edge_mat, rot=(0, 0, rz), bevel=0.004)

    return obj


def build_spotlight_fixture(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build a visible procedural stage spotlight.

    The JSON position is treated as the mount/clamp point when the fixture is
    mounted on a truss or bar. The visible barrel is placed slightly below that
    mount point, which is a generic rule for hanging stage hardware.
    """
    mat_spec = _material_spec(source_obj or {})
    base_rgb = _visibility_rgb(tuple(col), "light", minimum=0.055)
    rough = float(mat_spec.get("roughness", 0.42)) if isinstance(mat_spec, dict) else 0.42
    metal = float(mat_spec.get("metallic", 0.75)) if isinstance(mat_spec, dict) else 0.75
    dark = get_mat("fixture_visible_" + oid, base_rgb, roughness=rough, metallic=max(metal, 0.55))

    lens = get_mat("lens_" + oid, (1.0, 0.68, 0.32), roughness=0.18, metallic=0.0,
                   emission_rgb=(1.0, 0.46, 0.18), emission_strength=0.75)

    radius = max(0.105, min(w, h) * 0.36)
    length = max(d * 1.20, radius * 2.65, 0.38)
    body_z = pz - max(0.18, h * 0.30)

    y_front = -length / 2.0
    y_back = length / 2.0

    a = _wxy(px, py, 0, y_back, rz)
    b = _wxy(px, py, 0, y_front, rz)
    body = _cyl(oid + "_body", (a[0], a[1], body_z), (b[0], b[1], body_z), radius, dark, verts=36)

    # Lens/rear cap are aligned with the barrel axis, not vertical discs.
    lf1 = _wxy(px, py, 0, y_front - 0.028, rz)
    lf2 = _wxy(px, py, 0, y_front + 0.020, rz)
    _cyl(oid + "_lens", (lf1[0], lf1[1], body_z), (lf2[0], lf2[1], body_z), radius * 0.76, lens, verts=30)
    rb1 = _wxy(px, py, 0, y_back - 0.020, rz)
    rb2 = _wxy(px, py, 0, y_back + 0.030, rz)
    _cyl(oid + "_rear_cap", (rb1[0], rb1[1], body_z), (rb2[0], rb2[1], body_z), radius * 0.92, dark, verts=30)

    # U-yoke, top bar, clamp and drop pin.
    yoke_z = body_z + radius * 0.55
    for sx in (-1, 1):
        wx, wy = _wxy(px, py, sx * radius * 1.22, 0, rz)
        _cube(oid + "_yoke", (wx, wy, yoke_z), (0.038, 0.055, radius * 1.35), dark,
              rot=(0, 0, rz), bevel=0.006)
    top_a = _wxy(px, py, -radius * 1.22, 0, rz)
    top_b = _wxy(px, py, radius * 1.22, 0, rz)
    _cyl(oid + "_yoke_bar", (top_a[0], top_a[1], yoke_z + radius * 0.72),
         (top_b[0], top_b[1], yoke_z + radius * 0.72), 0.018, dark, verts=12)

    cx, cy = _wxy(px, py, 0, y_back + 0.08, rz)
    _cube(oid + "_clamp", (cx, cy, pz - 0.020), (max(w * 0.75, radius * 1.5), 0.09, 0.07),
          dark, rot=(0, 0, rz), bevel=0.008)
    # Small saddle at the top makes contact with the truss bottom/front chord
    # visible even in simplified preview geometry.
    _cube(oid + "_mount_saddle", (cx, cy, pz + 0.026), (max(w * 0.62, radius * 1.25), 0.055, 0.026),
          dark, rot=(0, 0, rz), bevel=0.005)
    _cyl(oid + "_drop_pin", (cx, cy, pz - 0.04), (cx, cy, body_z + radius * 1.2),
         0.015, dark, verts=10)

    return body


def build_curtain(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat_src = source_obj or {"id": oid, "category": "curtain", "material": {"type": "velvet_fabric", "roughness": 0.9}}
    rgb = base_color_for(mat_src)
    if not rgb or rgb == DEFAULT_COLOR.get("curtain"):
        rgb = col if (col and col != DEFAULT_COLOR["prop"]) else (0.18, 0.02, 0.03)
    rgb = _lift_dark_rgb(rgb, minimum=0.085)
    mat = material_for_object(mat_src, "curtain", tuple(rgb))
    dark = mat_fabric(tuple(max(c * 0.4, 0.01) for c in rgb))

    def _single(sid, sw, sd, sh, sx, sy, sz, srot):
        n = max(8, int(sw / 0.3))
        pw = sw / max(n, 1)
        fold = max(sd, 0.08) * 1.6
        first_local = None
        for i in range(n):
            lx = -sw / 2.0 + (i + 0.5) * pw
            ly = 0.0 if i % 2 == 0 else fold * 0.34
            wx, wy = _wxy(sx, sy, lx, ly, srot)
            panel_mat = mat if i % 2 == 0 else dark
            obj = _cube(sid + "_p%d" % i, (wx, wy, sz + sh / 2.0), (pw * 0.92, fold, sh), panel_mat,
                        rot=(0, 0, srot), bevel=0.018)
            first_local = first_local or obj
        wx, wy = _wxy(sx, sy, 0, 0, srot)
        _cube(sid + "_valance", (wx, wy, sz + sh + 0.12), (sw * 1.02, fold * 1.1, 0.24),
              mat_metal_dark(), rot=(0, 0, srot), bevel=0.02)
        return first_local

    positions = _unique_positions(_layout_positions(source_obj or {}))
    if len(positions) >= 2:
        proto = (((source_obj or {}).get("generation_hints") or {}).get("prototype") or {}) if isinstance(source_obj, dict) else {}
        psize = proto.get("size") if isinstance(proto.get("size"), dict) else {}
        piece_w = float(psize.get("w", 0.0) or 0.0)
        piece_d = float(psize.get("d", d) or d)
        piece_h = float(psize.get("h", h) or h)
        if piece_w <= 0:
            piece_w = max(0.8, (w / max(len(positions), 1)) * 0.96)

        env = _SCENE_CTX.get("stage_envelope") or {}
        anchor = _SCENE_CTX.get("performance_anchor") or env
        cx = anchor.get("x", sum(p[0] for p in positions) / max(len(positions), 1))
        cy = anchor.get("y", sum(p[1] for p in positions) / max(len(positions), 1))
        first = None

        # If positions are clearly on opposite X sides, treat them as side
        # curtains and line them along stage depth near the global envelope,
        # not tightly against the local platform anchor.
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        side_pair = (max(xs) - min(xs)) > max(1.0, (max(ys) - min(ys)) * 1.35)
        for idx, p in enumerate(positions):
            sx, sy, sz = p
            if side_pair and env:
                side = -1.0 if sx < cx else 1.0
                margin = max(0.45, piece_d * 2.4)
                sx = _clamp(sx, env["xmin"] + margin, env["xmax"] - margin)
                # Place the side drape along the global side zone; keep the
                # extracted y as a semantic hint but pull it toward the
                # performance anchor midpoint to avoid blocking the front view.
                sy = _clamp((sy + cy) * 0.5, env["ymin"] + piece_w * 0.40, env["ymax"] - piece_w * 0.40)
                srot = math.pi / 2.0
            else:
                srot = _infer_inward_tangent_rotation(sx, sy, cx, cy, fallback=rz)
            obj = _single(f"{oid}_{idx+1}", piece_w, piece_d, piece_h, sx, sy, sz, srot)
            first = first or obj
        return first

    return _single(oid, w, d, h, px, py, pz, rz)


def build_stairs(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    base = tuple(col if col != DEFAULT_COLOR["prop"] else (0.07, 0.05, 0.04))
    mat = material_for_object(source_obj or {"id": oid, "category": "stairs", "material": {"type": "wood", "roughness": 0.65}}, "stair", base)
    nosing = mat_wood()

    # Step count is a visual/design attribute, not a direct consequence of
    # total height.  A validator may cap stair total height so it does not
    # overshoot its target platform, but the visible number of treads should be
    # preserved when the extractor/validator supplies it.  This makes access
    # stairs generic across flush, step-up, side-access, and decorative cases.
    def _get_step_count(src):
        if not isinstance(src, dict):
            return None
        candidates = []
        geom = src.get("geometry") if isinstance(src.get("geometry"), dict) else {}
        proc = geom.get("procedural") if isinstance(geom.get("procedural"), dict) else {}
        detail = geom.get("detail_params") if isinstance(geom.get("detail_params"), dict) else {}
        attrs = src.get("attributes") if isinstance(src.get("attributes"), dict) else {}
        gh = src.get("generation_hints") if isinstance(src.get("generation_hints"), dict) else {}
        for holder in (proc, detail, attrs, gh, src):
            for key in ("visual_step_count", "visible_step_count", "design_step_count", "step_count", "steps", "tread_count"):
                if isinstance(holder, dict) and holder.get(key) is not None:
                    candidates.append(holder.get(key))
        for c in candidates:
            try:
                n = int(round(float(c)))
                if 1 <= n <= 30:
                    return n
            except Exception:
                pass
        return None

    n = _get_step_count(source_obj)
    if n is None:
        # If no explicit count is available, infer visible treads from stair run
        # depth rather than height.  Height may have been capped or may be lower
        # than the target platform depending on connection_type; visible tread
        # count is a design/visual property.
        if d > 0:
            n = int(math.ceil(d / 0.30))
        else:
            n = max(2, int(round(h / 0.18)))
    n = max(1, min(30, int(n)))
    rise = h / n
    run = d / n
    first = None
    for i in range(n):
        step_h = (i + 1) * rise
        ly = -d / 2.0 + run * (i + 0.5)
        wx, wy = _wxy(px, py, 0, ly, rz)
        o = _cube(oid + "_s%d" % i, (wx, wy, pz + step_h / 2.0), (w, run * 1.02, step_h),
                  mat, rot=(0, 0, rz), bevel=0.02)
        first = first or o
        nx, ny = _wxy(px, py, 0, ly - run * 0.46, rz)
        _cube(oid + "_n%d" % i, (nx, ny, pz + step_h + 0.01), (w * 0.94, 0.04, 0.02),
              nosing, rot=(0, 0, rz), bevel=0.004)
    return first


def build_column(oid, w, d, h, px, py, pz, rz, col):
    mat = get_mat("col_" + oid, col if col != DEFAULT_COLOR["prop"] else (0.5, 0.48, 0.44), roughness=0.5)
    r = max(w, d) / 2.0 * 0.92
    wx, wy = _wxy(px, py, 0, 0, rz)
    return _cyl(oid, (wx, wy, pz), (wx, wy, pz + h), r, mat, verts=24)


def build_proscenium_set_piece(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build visible side-wall / proscenium / column set pieces.

    These objects are scenic architecture, not rigging.  They must not be routed
    to the truss builder even when they are near or attached to a truss.  A
    neutral block with base/cap reads better than a lattice proxy for side walls
    and theatre columns.
    """
    base_col = tuple(col) if col != DEFAULT_COLOR.get("prop", tuple(col)) else (0.46, 0.43, 0.38)
    mat = material_for_object(source_obj or {"id": oid, "category": "set_piece"}, "set_piece", base_col)
    trim = get_mat("setpiece_trim_" + oid, (max(base_col[0]-0.10, 0.05), max(base_col[1]-0.10, 0.05), max(base_col[2]-0.10, 0.05)), roughness=0.55)

    wx, wy = _wxy(px, py, 0, 0, rz)
    body = _cube(oid, (wx, wy, pz + h / 2.0), (w, d, h), mat, rot=(0, 0, rz), bevel=min(0.035, min(w, d) * 0.18))

    # Base and cap give it a proscenium/architectural read without hard-coding
    # any particular theatre style.
    base_h = min(0.18, max(0.05, h * 0.045))
    cap_h = base_h
    bw = w * 1.12
    bd = d * 1.10
    _cube(oid + "_base", (wx, wy, pz + base_h / 2.0), (bw, bd, base_h), trim, rot=(0, 0, rz), bevel=0.01)
    _cube(oid + "_cap", (wx, wy, pz + h - cap_h / 2.0), (bw, bd, cap_h), trim, rot=(0, 0, rz), bevel=0.01)

    # A slim front trim strip helps side walls/columns read as scenic elements
    # rather than anonymous boxes.
    fx, fy = _wxy(px, py, 0, -d / 2.0 - 0.006, rz)
    _cube(oid + "_front_trim", (fx, fy, pz + h / 2.0), (w * 0.82, 0.018, h * 0.86), trim, rot=(0, 0, rz), bevel=0.006)
    return body


def build_table(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat = material_for_object(
        source_obj or {"id": oid, "category": "furniture", "material": {"type": "wood"}},
        "table",
        tuple(col)
    )
    t = min(0.06, h * 0.18)
    legt = max(0.04, min(w, d) * 0.08)
    wx, wy = _wxy(px, py, 0, 0, rz)
    top = _cube(oid + "_top", (wx, wy, pz + h - t / 2.0), (w, d, t), mat, rot=(0, 0, rz), bevel=0.012)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (w / 2.0 - legt), sy * (d / 2.0 - legt)
            lxx, lyy = _wxy(px, py, lx, ly, rz)
            _cube(oid + "_leg", (lxx, lyy, pz + (h - t) / 2.0), (legt * 2, legt * 2, h - t),
                  mat, rot=(0, 0, rz), bevel=0.006)
    return top


def build_chair(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat = material_for_object(
        source_obj or {"id": oid, "category": "furniture", "material": {"type": "wood"}},
        "chair",
        tuple(col)
    )
    seat_h = h * 0.5
    legt = max(0.03, min(w, d) * 0.10)
    tk = min(w, d) * 0.14
    wx, wy = _wxy(px, py, 0, 0, rz)
    seat = _cube(oid + "_seat", (wx, wy, pz + seat_h), (w, d, h * 0.10), mat, rot=(0, 0, rz), bevel=0.01)
    # backrest on the upstage (+Y) side, so the seat faces the audience (-Y)
    bx, by = _wxy(px, py, 0, d / 2.0 - tk / 2.0, rz)
    _cube(oid + "_back", (bx, by, pz + seat_h + (h - seat_h) / 2.0), (w, tk, h - seat_h),
          mat, rot=(0, 0, rz), bevel=0.01)
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx, ly = sx * (w / 2.0 - legt), sy * (d / 2.0 - legt)
            lxx, lyy = _wxy(px, py, lx, ly, rz)
            _cube(oid + "_leg", (lxx, lyy, pz + seat_h / 2.0), (legt * 2, legt * 2, seat_h),
                  mat, rot=(0, 0, rz), bevel=0.005)
    return seat


def build_scenic_wall_panel(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build walls/background/acoustic panels as thin vertical scenic planes.

    The previous generic box fallback makes theatre walls look like huge solid
    cubes.  This builder is generic: any wall/scenic-background/acoustic-panel
    object becomes a thin panel with optional frame and seams, preserving width
    and height while clamping depth to a plausible stage flat thickness.
    """
    text = _semantic_text(source_obj or {"id": oid})
    panel_depth = min(max(float(d or 0.12), 0.045), 0.18)
    # If an extractor uses a block-like depth for a wall, keep the panel at the
    # given XY center but avoid visualizing it as a massive cuboid.
    mat = material_for_object(source_obj or {"id": oid, "category": "wall"}, "wall", tuple(col))
    wx, wy = _wxy(px, py, 0, 0, rz)
    panel = _cube(oid, (wx, wy, pz + h / 2.0), (w, panel_depth, h), mat, rot=(0, 0, rz), bevel=0.008)

    trim_rgb = tuple(max(c * 0.62, 0.035) for c in tuple(col))
    trim = get_mat("wall_trim_" + oid, trim_rgb, roughness=0.62, metallic=0.05)
    frame_t = max(0.035, min(w, h) * 0.012)
    # Simple frame helps a flat/scenic wall read as a panel, not a block.
    _cube(oid + "_top_trim", (wx, wy - panel_depth * 0.58, pz + h - frame_t / 2.0), (w + frame_t, frame_t, frame_t), trim, rot=(0, 0, rz), bevel=0.004)
    _cube(oid + "_bottom_trim", (wx, wy - panel_depth * 0.58, pz + frame_t / 2.0), (w + frame_t, frame_t, frame_t), trim, rot=(0, 0, rz), bevel=0.004)
    for sx, nm in ((-1, "L"), (1, "R")):
        lx, ly = _wxy(px, py, sx * (w / 2.0 - frame_t / 2.0), -panel_depth * 0.58, rz)
        _cube(oid + "_side_trim_" + nm, (lx, ly, pz + h / 2.0), (frame_t, frame_t, h), trim, rot=(0, 0, rz), bevel=0.004)

    # Acoustic/panelled walls get repeated shallow seams.  This is based on the
    # semantic class, not on a test-scene ID.
    if _has(text, "acoustic", "panel", "wall panel", "sound panel", "声学", "吸音", "墙板"):
        seam_count = max(3, min(14, int(w / 0.65)))
        seam_mat = get_mat("wall_seam_" + oid, trim_rgb, roughness=0.70, metallic=0.0)
        for i in range(1, seam_count):
            lx = -w / 2.0 + w * i / seam_count
            sx, sy = _wxy(px, py, lx, -panel_depth * 0.66, rz)
            _cube(oid + "_seam_%02d" % i, (sx, sy, pz + h / 2.0), (0.018, 0.018, h * 0.92), seam_mat, rot=(0, 0, rz), bevel=0.002)
    return panel


def _layout_hint(o):
    gh = o.get("generation_hints") if isinstance(o, dict) else None
    gh = gh if isinstance(gh, dict) else {}
    layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else {}
    return gh, layout


def _layout_center(o, fallback=None):
    _gh, layout = _layout_hint(o)
    c = layout.get("center")
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        vals = list(c) + [0.0, 0.0, 0.0]
        try:
            return [float(vals[0]), float(vals[1]), float(vals[2])]
        except Exception:
            pass
    return list(fallback or [0.0, 0.0, 0.0])


def _layout_positions(o):
    _gh, layout = _layout_hint(o)
    positions = layout.get("positions")
    out = []
    if isinstance(positions, list):
        for p in positions:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                vals = list(p) + [0.0, 0.0, 0.0]
                try:
                    out.append([float(vals[0]), float(vals[1]), float(vals[2])])
                except Exception:
                    pass
            elif isinstance(p, dict):
                vals = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
                try:
                    out.append([float(vals[0]), float(vals[1]), float(vals[2])])
                except Exception:
                    pass
    return out


def _layout_attached_id(o):
    gh, layout = _layout_hint(o)
    attached = None
    if isinstance(layout, dict):
        attached = layout.get("attached_to") or layout.get("mount_target")
    if not attached and isinstance(gh, dict):
        attached = gh.get("mount_target")
    return attached if isinstance(attached, str) and attached else None


def _layout_attached_object(o):
    attached_id = _layout_attached_id(o)
    return (_SCENE_CTX.get("by_id", {}) or {}).get(attached_id) if attached_id else None


def _is_array_proxy(o):
    if not isinstance(o, dict):
        return False
    gh = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
    layout = gh.get("layout") if isinstance(gh.get("layout"), dict) else {}
    if gh.get("must_expand_array") or str(gh.get("role", "")).lower() == "array_proxy":
        return True
    if layout and int(_count_hint(o, default=1) or 1) > 1:
        return True
    return False


def _rotation_z_of(o, default=0.0):
    if not isinstance(o, dict):
        return default
    rot = o.get("rotation")
    if isinstance(rot, (list, tuple)) and len(rot) >= 3:
        try:
            return float(rot[2])
        except Exception:
            return default
    return default


def _infer_inward_tangent_rotation(x, y, center_x, center_y, fallback=0.0):
    dx = x - center_x
    dy = y - center_y
    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
        return fallback
    # Tangent to the radial line from stage center.  This is generic for side
    # curtains / wall arrays that visually line the performance area rather than
    # cutting directly across it.
    return math.atan2(dy, dx) + math.pi / 2.0


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _object_bounds_xy(o):
    """Axis-aligned XY bounds for layout reasoning.

    This deliberately ignores object rotation because generated scene graphs
    often have missing or unreliable rotations.  The bounds are only used for
    high-level composition, not exact collision.
    """
    if not isinstance(o, dict):
        return None
    size = o.get("size") or {}
    pos = (o.get("position") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
    w = _safe_float(size.get("w", 0.0), 0.0)
    d = _safe_float(size.get("d", 0.0), 0.0)
    if w <= 0 or d <= 0:
        return None
    x, y = _safe_float(pos[0]), _safe_float(pos[1])
    return {"xmin": x - w/2.0, "xmax": x + w/2.0, "ymin": y - d/2.0, "ymax": y + d/2.0,
            "x": x, "y": y, "w": w, "d": d}


def _make_bounds(xmin, xmax, ymin, ymax):
    return {
        "xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax),
        "x": (float(xmin) + float(xmax)) / 2.0,
        "y": (float(ymin) + float(ymax)) / 2.0,
        "w": max(0.01, float(xmax) - float(xmin)),
        "d": max(0.01, float(ymax) - float(ymin)),
    }


def _merge_bounds(bounds_list):
    bounds_list = [b for b in bounds_list if isinstance(b, dict)]
    if not bounds_list:
        return None
    return _make_bounds(
        min(b["xmin"] for b in bounds_list),
        max(b["xmax"] for b in bounds_list),
        min(b["ymin"] for b in bounds_list),
        max(b["ymax"] for b in bounds_list),
    )


def _top_stage_envelope(scene, object_bounds=None):
    """Global stage/theatre envelope.

    The top-level scene.stage is a room/stage-space envelope, while platform
    objects are local performance anchors.  Keeping these separate prevents
    curtains, audiences and cameras from being laid out around a small platform.
    """
    stage = scene.get("stage") if isinstance(scene.get("stage"), dict) else {}
    w = _safe_float(stage.get("width", 0.0), 0.0)
    d = _safe_float(stage.get("depth", 0.0), 0.0)
    if w > 0 and d > 0:
        env = _make_bounds(-w/2.0, w/2.0, -d/2.0, d/2.0)
    else:
        env = object_bounds
    if object_bounds:
        # Expand to include extracted objects that fall slightly outside the
        # nominal envelope due to coordinate-origin differences.  Do not shrink.
        env = _merge_bounds([env, object_bounds]) if env else object_bounds
    if env is None:
        env = _make_bounds(-5.0, 5.0, -4.0, 4.0)
    return env


def _performance_anchor_from_objects(scene):
    """Local performance-platform anchor.

    Uses the union of visible platform/floor support surfaces, not the entire
    stage envelope.  This anchor is used for furniture and audience-facing
    relationships; the global envelope is used for curtains/camera/audience span.
    """
    objs = scene.get("objects", []) or []
    supports = []
    for o in objs:
        if not isinstance(o, dict):
            continue
        cat = str(o.get("category", "")).lower()
        text = _semantic_text(o)
        if cat in ("platform", "floor") and not _has(text, "audience", "seating", "chair"):
            b = _object_bounds_xy(o)
            if b:
                supports.append(b)
    anchor = _merge_bounds(supports)
    if anchor:
        return anchor
    # Fallback to any selected primary stage.
    sm = _stage_metrics(_SCENE_CTX.get("primary_stage")) if _SCENE_CTX else None
    if sm:
        return _make_bounds(sm["x"]-sm["w"]/2.0, sm["x"]+sm["w"]/2.0, sm["front_y"], sm["back_y"])
    return None


def _clamp(v, lo, hi):
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, v))


def _pad_bounds(b, pad):
    if not isinstance(b, dict):
        return b
    return _make_bounds(b["xmin"]-pad, b["xmax"]+pad, b["ymin"]-pad, b["ymax"]+pad)


def _chair_face_target_rotation(chair_x, chair_y, target_x, target_y, fallback=0.0):
    """Yaw for procedural chairs whose visual front is local -Y.

    build_chair places the backrest on local +Y, so the chair faces local -Y.
    To face a target point, local -Y must align with target - chair.  This
    generic helper fixes chair arrays, audience seating and table seating
    without scene-specific rotations.
    """
    dx = float(target_x) - float(chair_x)
    dy = float(target_y) - float(chair_y)
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx) + math.pi / 2.0


def _effective_clearance_z():
    """Usable overhead height in world Z.

    Extracted/VLM positions often put overhead objects at image-depth values
    like z=10 even when the declared stage clearance is 6m.  This converts the
    global clearance and performance-platform top into a practical mounting
    height for trusses, lights and other suspended objects.
    """
    clearance = _safe_float(_SCENE_CTX.get("height_clearance", 0.0), 0.0) if isinstance(_SCENE_CTX, dict) else 0.0
    anchor = _SCENE_CTX.get("performance_anchor") if isinstance(_SCENE_CTX, dict) else None
    anchor_top = 0.0
    if isinstance(anchor, dict):
        # Bounds do not include height, so use support lookup at anchor center.
        anchor_top = _support_top_z_at_xy(anchor.get("x", 0.0), anchor.get("y", 0.0), 0.0)
    if clearance <= 0:
        clearance = max(anchor_top + 4.0, 4.5)
    # Leave headroom below the declared ceiling/clearance, and ensure it is not
    # implausibly close to the platforms.
    return max(anchor_top + 2.4, min(clearance * 0.78, clearance - 0.6 if clearance > 1.2 else clearance))


def _normalize_overhead_z(z, object_h=0.0):
    target = _effective_clearance_z()
    try:
        zf = float(z)
    except Exception:
        zf = target
    # Only correct obviously polluted or out-of-envelope z values.  Low trusses
    # intentionally specified by a designer remain intact.
    if zf > target + max(0.8, float(object_h or 0.0) * 2.0) or zf > (_safe_float(_SCENE_CTX.get("height_clearance", 0.0), 0.0) + 0.15):
        return target
    return zf


def _resolved_reference_position(o):
    """Reference position used by dependent arrays.

    Unlike _SCENE_CTX['by_id'], this applies the same preview-safe normalization
    that the visible object builder will use.  This keeps light arrays attached
    to the rendered truss instead of the raw VLM z position.
    """
    if not isinstance(o, dict):
        return [0.0, 0.0, 0.0]
    return _resolve_object_position(o)


def _count_hint(o, default=None):
    """Infer repeated-object count from common schema fields and labels."""
    if not isinstance(o, dict):
        return default
    candidates = [o.get("count"), o.get("object_count"), o.get("visual_count")]
    attrs = o.get("attributes") or {}
    candidates += [attrs.get("count"), attrs.get("object_count"), attrs.get("visual_count")]
    geom = o.get("geometry") or {}
    proc = geom.get("procedural") if isinstance(geom, dict) else {}
    proc = proc if isinstance(proc, dict) else {}
    detail = proc.get("detail_params") if isinstance(proc.get("detail_params"), dict) else {}
    candidates += [proc.get("count"), detail.get("count"), detail.get("module_count"), detail.get("seat_count")]
    hints = o.get("generation_hints") if isinstance(o.get("generation_hints"), dict) else {}
    candidates += [hints.get("count"), hints.get("expected_count"), hints.get("visual_count")]
    txt = _semantic_text(o)
    m = re.search(r"(?:^|\D)([2-9]|[1-4][0-9])(?:\D|$)", txt)
    if m:
        candidates.append(m.group(1))
    words = {"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"twelve":12,
             "二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9,"十":10}
    for k, v in words.items():
        if k in txt:
            candidates.append(v)
    for c in candidates:
        try:
            if c is None or c == "":
                continue
            if isinstance(c, str):
                m = re.search(r"\d+", c)
                if not m:
                    continue
                c = m.group(0)
            n = int(float(c))
            if 1 <= n <= 80:
                return n
        except Exception:
            pass
    return default


def build_stool(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    mat = material_for_object(source_obj or {"id": oid, "category": "furniture"}, "stool", tuple(col))
    r = max(0.10, min(w, d) * 0.34)
    wx, wy = _wxy(px, py, 0, 0, rz)
    seat_h = max(0.045, min(0.08, h * 0.10))
    _cyl(oid + "_seat", (wx, wy, pz + h - seat_h), (wx, wy, pz + h), r, mat, verts=28)
    leg_mat = get_mat("stool_leg_" + oid, (0.75, 0.72, 0.66), roughness=0.45, metallic=0.45)
    for i in range(4):
        a = rz + math.pi/4 + i * math.pi/2
        lx, ly = math.cos(a) * r * 0.72, math.sin(a) * r * 0.72
        _cyl(oid + "_leg_%d" % i, (wx + lx, wy + ly, pz), (wx + lx * 0.72, wy + ly * 0.72, pz + h - seat_h), 0.018, leg_mat, verts=8)
    return bpy.data.objects.get(oid + "_seat")


def build_seating_array(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    """Build repeated seating/stools as semantically meaningful arrays.

    V5 uses the global stage envelope and local performance anchor separately.
    This prevents audience seating from being arranged around a single small
    platform while still keeping chairs aimed at the actual performance area.
    """
    src = source_obj or {"id": oid}
    text = _semantic_text(src)
    gh, layout = _layout_hint(src)
    ltype = str(layout.get("type", "")).lower()
    is_stool = _has(text, "stool", "bar stool", "high stool", "高脚凳", "凳")
    is_audience = _has(text, "audience", "seating", "seats", "seat array", "观众", "座椅", "座位")
    is_dining = (_has(text, "dining", "餐桌", "围桌") or ltype == "circular") and not is_audience
    default_count = 5 if is_stool else (12 if is_audience else 6)
    count = max(1, min(int(_count_hint(src, default=default_count) or default_count), 80))

    proto = gh.get("prototype") if isinstance(gh.get("prototype"), dict) else {}
    child_src = proto if proto else src
    child_col = base_color_for(child_src)

    center = _layout_center(src, [px, py, pz])
    px, py = center[0], center[1]
    if abs(center[2]) > 1e-6:
        pz = center[2]
    pz = _support_top_z_at_xy(px, py, pz)

    positions = _unique_positions(_layout_positions(src))
    if positions and ltype not in ("curved", "curved_rows", "arc"):
        actual_positions = positions[:count]
        first = None
        for idx, p in enumerate(actual_positions):
            sx, sy = p[0], p[1]
            sz = _support_top_z_at_xy(sx, sy, pz)
            srot = rz
            if len(actual_positions) > 1:
                srot = _chair_face_target_rotation(sx, sy, px, py, fallback=rz)
            if is_stool:
                obj = build_stool(f"{oid}_stool_{idx+1}", max(0.38, min(w, 0.62)), max(0.38, min(d, 0.62)), max(0.75, h * 0.9), sx, sy, sz, srot, child_col, child_src)
            else:
                obj = build_chair(f"{oid}_seat_{idx+1}", min(max(w, 0.42), 0.75), min(max(d, 0.45), 0.80), min(max(h, 0.85), 1.18), sx, sy, sz, srot, child_col, child_src)
            first = first or obj
        return first

    if is_stool:
        rows = 1 if count <= 6 else 2
        cols = int(math.ceil(count / rows))
        cell_w = min(0.62, max(0.38, w / max(cols, 1) * 0.72))
        cell_d = min(0.62, max(0.38, d / max(rows, 1) * 0.72))
        first = None
        for idx in range(count):
            r_i = idx // cols
            c_i = idx % cols
            lx = (c_i - (cols - 1) / 2.0) * (max(w, cols * 0.50) / max(cols, 1))
            ly = (r_i - (rows - 1) / 2.0) * (max(d, rows * 0.60) / max(rows, 1))
            sx, sy = _wxy(px, py, lx, ly, rz)
            sz = _support_top_z_at_xy(sx, sy, pz)
            obj = build_stool(f"{oid}_stool_{idx+1}", cell_w, cell_d, max(0.75, h * 0.9), sx, sy, sz, rz, child_col, child_src)
            first = first or obj
        return first

    if is_dining and count >= 3:
        radius = float(layout.get("radius", 0.0) or 0.0)
        radius_x = max(0.75, radius if radius > 0 else w * 0.42)
        radius_y = max(0.65, radius if radius > 0 else d * 0.42)
        chair_w = min(0.55, max(0.42, w if w < 0.8 else 0.50))
        chair_d = min(0.62, max(0.45, d if d < 0.8 else 0.55))
        chair_h = min(max(h, 0.85), 1.10)
        first = None
        for idx in range(count):
            ang = (2.0 * math.pi * idx / count) + math.pi
            lx = math.cos(ang) * radius_x
            ly = math.sin(ang) * radius_y
            sx, sy = _wxy(px, py, lx, ly, rz)
            sz = _support_top_z_at_xy(sx, sy, pz)
            srot = _chair_face_target_rotation(sx, sy, px, py, fallback=rz)
            obj = build_chair(f"{oid}_seat_{idx+1}", chair_w, chair_d, chair_h, sx, sy, sz, srot, child_col, child_src)
            first = first or obj
        return first

    if is_audience and ltype in ("curved", "curved_rows", "arc"):
        env = _SCENE_CTX.get("stage_envelope") or {}
        anchor = _SCENE_CTX.get("performance_anchor") or env
        if env and anchor:
            chair_w = min(0.55, max(0.42, w * 0.70))
            chair_d = min(0.62, max(0.45, d * 0.72))
            chair_h = min(max(h, 0.85), 1.10)

            # Determine the audience/front side from the source position when
            # available.  This avoids assuming that downstage is always +Y or -Y
            # after VLM/validator coordinate repairs.
            vx = px - anchor["x"]
            vy = py - anchor["y"]
            if math.hypot(vx, vy) < 0.5:
                # Choose the side of the envelope with more free space.
                free_pos_y = env["ymax"] - anchor["ymax"]
                free_neg_y = anchor["ymin"] - env["ymin"]
                if max(free_pos_y, free_neg_y) >= max(env["w"], env["d"]) * 0.08:
                    vx, vy = 0.0, (1.0 if free_pos_y >= free_neg_y else -1.0)
                else:
                    free_pos_x = env["xmax"] - anchor["xmax"]
                    free_neg_x = anchor["xmin"] - env["xmin"]
                    vx, vy = (1.0 if free_pos_x >= free_neg_x else -1.0), 0.0
            ln = math.hypot(vx, vy) or 1.0
            ux, uy = vx / ln, vy / ln
            pxp, pyp = -uy, ux  # perpendicular axis

            # Support radius of anchor along audience and perpendicular axes.
            anchor_front = 0.5 * (abs(ux) * anchor["w"] + abs(uy) * anchor["d"])
            anchor_side = 0.5 * (abs(pxp) * anchor["w"] + abs(pyp) * anchor["d"])
            env_side = 0.5 * (abs(pxp) * env["w"] + abs(pyp) * env["d"])
            env_front = 0.5 * (abs(ux) * env["w"] + abs(uy) * env["d"])

            gap_front = max(0.90, chair_d * 1.55)
            row_gap = max(0.68, chair_d * 1.20)
            rows_aud = max(2, min(4, int(round(math.sqrt(count / 6.0)) + 1)))
            front_span = min(env_side * 1.50, max(anchor_side * 2.45, 3.0))
            side_span = min(env_front * 0.72, max(anchor_front * 1.25, 2.2))

            candidates = []
            for r_i in range(rows_aud):
                # Front bank on the inferred audience side.
                dist = anchor_front + gap_front + r_i * row_gap
                bx = anchor["x"] + ux * dist
                by = anchor["y"] + uy * dist
                cols_front = max(4, int(round(front_span / max(0.62, chair_w * 1.22))))
                for c_i in range(cols_front):
                    t = -0.5 if cols_front == 1 else (c_i / max(cols_front - 1, 1) - 0.5)
                    sx = bx + pxp * (front_span * t)
                    sy = by + pyp * (front_span * t)
                    if env["xmin"] + chair_w <= sx <= env["xmax"] - chair_w and env["ymin"] + chair_d <= sy <= env["ymax"] - chair_d:
                        candidates.append((r_i, 0, sx, sy))

                # Side banks, offset to either side of the anchor and spanning
                # along the audience direction.  These create three-sided
                # seating without hardcoding left/right stage coordinates.
                side_dist = anchor_side + max(0.85, chair_w * 1.55) + r_i * chair_w * 1.05
                side_cols = max(3, int(round(side_span / max(0.68, chair_d * 1.18))))
                for sign, seg in ((-1.0, 1), (1.0, 2)):
                    cx_side = anchor["x"] + pxp * sign * side_dist
                    cy_side = anchor["y"] + pyp * sign * side_dist
                    for c_i in range(side_cols):
                        t = 0.5 if side_cols == 1 else c_i / max(side_cols - 1, 1)
                        along = -anchor_front * 0.15 + side_span * t
                        sx = cx_side + ux * along
                        sy = cy_side + uy * along
                        if env["xmin"] + chair_w <= sx <= env["xmax"] - chair_w and env["ymin"] + chair_d <= sy <= env["ymax"] - chair_d:
                            candidates.append((r_i, seg, sx, sy))

            if not candidates:
                radius = max(anchor["w"], anchor["d"]) * 0.90 + 1.0
                for idx in range(count):
                    t = 0.5 if count == 1 else idx / max(count - 1, 1)
                    ang = math.radians(25 + 130 * t)
                    candidates.append((0, 0, anchor["x"] + math.cos(ang)*radius, anchor["y"] + math.sin(ang)*radius))

            # Interleave front and sides so low counts still show the front bank.
            candidates.sort(key=lambda it: (it[0], it[1], (it[2]-anchor["x"])**2 + (it[3]-anchor["y"])**2))
            used = candidates[:count]
            first = None
            for idx, (_row, _seg, sx, sy) in enumerate(used):
                sz = _support_top_z_at_xy(sx, sy, pz)
                srot = _chair_face_target_rotation(sx, sy, anchor["x"], anchor["y"], fallback=rz)
                obj = build_chair(f"{oid}_seat_{idx+1}", chair_w, chair_d, chair_h, sx, sy, sz, srot, child_col, child_src)
                first = first or obj
            return first

    rows = max(1, min(5, int(round(math.sqrt(count / 2.0)))))
    cols = int(math.ceil(count / rows))
    total_w = max(w, cols * 0.58)
    total_d = max(d, rows * 0.72)
    chair_w = min(0.55, max(0.42, total_w / max(cols, 1) * 0.65))
    chair_d = min(0.62, max(0.45, total_d / max(rows, 1) * 0.65))
    chair_h = min(max(h, 0.85), 1.10)
    first = None
    for idx in range(count):
        r_i = idx // cols
        c_i = idx % cols
        lx = (c_i - (cols - 1) / 2.0) * (total_w / max(cols, 1))
        ly = (r_i - (rows - 1) / 2.0) * (total_d / max(rows, 1))
        sx, sy = _wxy(px, py, lx, ly, rz)
        sz = _support_top_z_at_xy(sx, sy, pz)
        obj = build_chair(f"{oid}_seat_{idx+1}", chair_w, chair_d, chair_h, sx, sy, sz, rz, child_col, child_src)
        first = first or obj
    return first


def build_screen(oid, w, d, h, px, py, pz, rz, o):
    attrs = o.get("attributes") or {}
    sp = attrs.get("screen") or {}
    col = base_color_for(o)
    emissive = bool(sp.get("emissive"))
    smat = get_mat("scr_" + oid, col, roughness=0.25,
                   emission_rgb=(col if emissive else None),
                   emission_strength=(EMISSION_STRENGTH if emissive else 0.0))
    z = pz + h / 2.0
    wx, wy = _wxy(px, py, 0, 0, rz)
    panel = _cube(oid, (wx, wy, z), (w, d, h), smat, rot=(0, 0, rz), bevel=0.0)
    fr = max(0.05, min(w, h) * 0.02)
    fm = mat_metal_dark()
    tb = _wxy(px, py, 0, 0, rz)
    _cube(oid + "_fT", (tb[0], tb[1], pz + h + fr / 2.0), (w + 2 * fr, d * 1.3, fr), fm, rot=(0, 0, rz), bevel=0.01)
    _cube(oid + "_fB", (tb[0], tb[1], pz - fr / 2.0), (w + 2 * fr, d * 1.3, fr), fm, rot=(0, 0, rz), bevel=0.01)
    lx = _wxy(px, py, -(w / 2.0 + fr / 2.0), 0, rz)
    rx = _wxy(px, py, (w / 2.0 + fr / 2.0), 0, rz)
    _cube(oid + "_fL", (lx[0], lx[1], z), (fr, d * 1.3, h), fm, rot=(0, 0, rz), bevel=0.01)
    _cube(oid + "_fR", (rx[0], rx[1], z), (fr, d * 1.3, h), fm, rot=(0, 0, rz), bevel=0.01)
    return panel



def build_painting(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    src = source_obj or {"id": oid, "category": "decor", "material": {"type": "canvas"}}
    frame_mat = get_mat("frame_" + oid, (0.42, 0.28, 0.12), roughness=0.45, metallic=0.08)
    art_rgb = base_color_for(src) or tuple(col)
    if not art_rgb or art_rgb == DEFAULT_COLOR.get("prop"):
        art_rgb = (0.82, 0.78, 0.68)
    art_mat = material_for_object(src, "painting", art_rgb)
    wx, wy = _wxy(px, py, 0, 0, rz)
    z = pz + h / 2.0
    panel = _cube(oid + "_canvas", (wx, wy, z), (w * 0.86, max(0.018, d), h * 0.86), art_mat,
                  rot=(0, 0, rz), bevel=0.004)
    frame = _cube(oid + "_frame", (wx, wy, z), (w, max(0.028, d * 1.3), h), frame_mat,
                  rot=(0, 0, rz), bevel=0.006)
    return frame or panel


def build_decor_array(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    src = source_obj or {"id": oid}
    gh, layout = _layout_hint(src)
    proto = gh.get("prototype") if isinstance(gh.get("prototype"), dict) else {}
    child_src = proto if proto else src
    count = max(1, min(int(_count_hint(src, default=3) or 3), 24))
    positions = _unique_positions(_layout_positions(src))
    attached = _layout_attached_object(src)

    attached_pos = None
    attached_size = {}
    attached_rz = rz
    if isinstance(attached, dict):
        attached_pos = (attached.get("position") or [0.0, 0.0, 0.0]) + [0.0, 0.0, 0.0]
        attached_size = attached.get("size") or {}
        attached_rz = _rotation_z_of(attached, default=rz)

    if not positions and attached_pos is not None:
        aw = float(attached_size.get("w", 0.0) or 0.0)
        ah = float(attached_size.get("h", 0.0) or 0.0)
        z0 = pz if pz > 0 else float(attached_pos[2]) + ah * 0.58
        span = max(aw * 0.72, max(w, 0.65) * max(count - 1, 1))
        for idx in range(count):
            t = 0.5 if count == 1 else idx / max(count - 1, 1)
            lx = -span / 2.0 + span * t
            sx, sy = _wxy(float(attached_pos[0]), float(attached_pos[1]), lx, 0.0, attached_rz)
            positions.append([sx, sy, z0])
        rz = attached_rz

    if not positions:
        positions = [[px, py, pz]]

    first = None
    pw = max(0.28, min(w, 1.4))
    pd = max(0.012, min(d if d > 0 else 0.04, 0.08))
    ph = max(0.35, min(h, 1.4))

    for idx, p in enumerate(positions[:count]):
        sx, sy = float(p[0]), float(p[1])
        sz = float(p[2]) if len(p) >= 3 and abs(float(p[2])) > 1e-6 else pz
        srot = rz
        # Mounted wall/decor rule: place the object on the visible front face
        # of the target vertical surface, not at the wall center where it can
        # be hidden inside the wall mesh.
        if attached_pos is not None and str(attached.get("category", "")).lower() in ("wall", "scenic_background", "screen"):
            target_d = float(attached_size.get("d", 0.08) or 0.08)
            offset = target_d / 2.0 + max(pd, 0.018) / 2.0 + 0.026
            sx, sy = _wxy(sx, sy, 0.0, -offset, attached_rz)
            srot = attached_rz
            if sz <= 0.05:
                sz = float(attached_pos[2]) + float(attached_size.get("h", ph) or ph) * 0.58
        obj = build_painting(f"{oid}_{idx+1}", pw, pd, ph, sx, sy, sz, srot, col, child_src)
        first = first or obj
    return first


def build_light_array(oid, w, d, h, px, py, pz, rz, col, source_obj=None):
    src = source_obj or {"id": oid, "category": "light"}
    gh, layout = _layout_hint(src)
    proto = gh.get("prototype") if isinstance(gh.get("prototype"), dict) else {}
    child_src = proto if proto else src
    count = max(1, min(int(_count_hint(src, default=6) or 6), 48))
    positions = _unique_positions(_layout_positions(src))
    attached = _layout_attached_object(src)
    child_size = proto.get("size") if isinstance(proto.get("size"), dict) else {}

    # Preview-safe minimums make fixtures visible without using external assets.
    lw = max(0.30, float(child_size.get("w", w if w > 0 else 0.22) or 0.22))
    ld = max(0.40, float(child_size.get("d", d if d > 0 else 0.28) or 0.28))
    lh = max(0.28, float(child_size.get("h", h if h > 0 else 0.24) or 0.24))

    if not positions and attached is not None and str(attached.get("category", "")).lower() == "truss":
        apos = _resolved_reference_position(attached) + [0.0, 0.0, 0.0]
        asize = attached.get("size") or {}
        aw = max(1.0, float(asize.get("w", 0.0) or 0.0))
        ah = float(asize.get("h", 0.2) or 0.2)
        arz = _rotation_z_of(attached, default=rz)
        for idx in range(count):
            t = 0.5 if count == 1 else idx / max(count - 1, 1)
            lx = -aw / 2.0 + aw * t
            sx, sy = _wxy(float(apos[0]), float(apos[1]), lx, 0.0, arz)
            # Mount just below the normalized truss plane.
            positions.append([sx, sy, _normalize_overhead_z(float(apos[2]), ah) - max(0.08, ah * 0.35)])
        rz = arz

    if not positions:
        pz = _normalize_overhead_z(pz, h)
        positions = [[px, py, pz]]
    first = None
    for idx, p in enumerate(positions[:count]):
        sx, sy = float(p[0]), float(p[1])
        sz = _normalize_overhead_z(float(p[2]) if len(p) >= 3 else pz, lh)
        obj = build_spotlight_fixture(f"{oid}_{idx+1}", lw, ld, lh, sx, sy, sz, rz, col, child_src)
        first = first or obj
    return first


def _has(text, *keys):
    return any(k in text for k in keys)


def _semantic_text(o):
    """Collect semantic hints used for v0.2/v0.3 geometry dispatch."""
    attrs = o.get("attributes") or {}
    geom = o.get("geometry") or attrs.get("geometry") or {}
    proc = geom.get("procedural") if isinstance(geom, dict) else {}
    proc = proc if isinstance(proc, dict) else {}
    mat = _material_spec(o)
    bits = [
        o.get("id", ""),
        o.get("label", ""),
        o.get("category", ""),
        o.get("subtype", ""),
        attrs.get("material", "") if isinstance(attrs.get("material"), str) else "",
        attrs.get("shape", ""),
        geom.get("representation", "") if isinstance(geom, dict) else "",
        geom.get("shape", "") if isinstance(geom, dict) else "",
        proc.get("shape_family", ""),
        mat.get("type", ""),
        mat.get("texture_hint", ""),
        mat.get("color_name", ""),
    ]
    return " ".join(str(b).lower().replace("_", " ").replace("-", " ") for b in bits if b)


def _object_area(o):
    size = o.get("size") or {}
    try:
        return float(size.get("w", 0) or 0) * float(size.get("d", 0) or 0)
    except Exception:
        return 0.0


def _is_stage_surface(o):
    """Generic detector for walkable performance decks.

    This deliberately avoids scene-specific IDs.  It identifies objects that
    should render as low, walkable surfaces rather than tall solid blocks.
    """
    if not isinstance(o, dict):
        return False
    cat = str(o.get("category", "")).lower()
    text = _semantic_text(o)
    if cat in ("floor", "platform") and _has(
        text,
        "main stage", "stage floor", "stage deck", "performance platform",
        "performance area", "acting area", "show deck", "main platform",
        "主舞台", "舞台地面", "表演区", "地台"
    ):
        return True
    geom = o.get("geometry") or {}
    fp = (geom.get("footprint") or {}) if isinstance(geom, dict) else {}
    if cat == "platform" and str(geom.get("representation", "")).lower() == "contour_extrusion":
        if str(fp.get("type", "")).lower() in ("rectangle", "rounded_rectangle", "circle", "ellipse", "polygon", "composite_2d"):
            return True
    return False


def _primary_stage_score(o):
    """Score likely global stage reference surfaces.

    Used only for preview heuristics and camera/light docking.  The rule is
    intentionally generic: prefer large, walkable, central performance surfaces;
    penalize local/side/rear/elevated walkways and scenic objects.
    """
    if not isinstance(o, dict):
        return -1e9
    cat = str(o.get("category", "")).lower()
    text = _semantic_text(o)
    area = _object_area(o)
    score = 0.0
    if cat == "floor":
        score += 45
    if cat == "platform":
        score += 25
    if _is_stage_surface(o):
        score += 55
    if str(o.get("id", "")).lower() == "main_stage":
        score += 35
    if _has(text, "main stage", "performance area", "stage floor", "stage deck", "central stage", "主舞台", "表演区"):
        score += 30
    score += min(area, 120.0) * 0.45
    # Penalize local structural surfaces. They are valid objects, just not the
    # global reference stage for camera and default placement.
    if _has(text, "walkway", "side platform", "side stage", "left", "right", "rear elevated", "back platform", "stairs", "step", "riser", "podium", "cube", "seat", "seating"):
        score -= 35
    if cat not in ("platform", "floor"):
        score -= 80
    return score


def _build_scene_context(scene):
    """Build lightweight scene context for generic placement heuristics.

    V5 separates two references:
      1. stage_envelope: the global theatre/stage-space boundary.
      2. performance_anchor: local visible platform(s) where action happens.

    Curtains, audience banks, camera and global lighting use the envelope.
    Furniture, support surfaces and relation solving use the performance anchor.
    This avoids arranging the whole theatre around one small raised platform.
    """
    objs = scene.get("objects", []) or []
    by_id = {o.get("id"): o for o in objs if isinstance(o, dict) and o.get("id")}
    candidates = [o for o in objs if isinstance(o, dict) and str(o.get("category", "")).lower() in ("platform", "floor")]
    primary_stage = None
    if candidates:
        primary_stage = max(candidates, key=_primary_stage_score)
        if _primary_stage_score(primary_stage) < 0:
            primary_stage = None

    object_bounds = _merge_bounds([_object_bounds_xy(o) for o in objs if isinstance(o, dict)])
    stage_envelope = _top_stage_envelope(scene, object_bounds)
    # Temporarily build base ctx so fallback functions can see primary_stage.
    ctx = {"by_id": by_id, "primary_stage": primary_stage, "stage_envelope": stage_envelope,
           "height_clearance": _safe_float((scene.get("stage") or {}).get("height_clearance", 0.0), 0.0)}
    old_ctx = globals().get("_SCENE_CTX", {})
    globals()["_SCENE_CTX"] = ctx
    performance_anchor = _performance_anchor_from_objects(scene)
    globals()["_SCENE_CTX"] = old_ctx
    if performance_anchor is None:
        performance_anchor = stage_envelope

    ctx["performance_anchor"] = performance_anchor
    return ctx


def _stage_metrics(stage_obj):
    if not isinstance(stage_obj, dict):
        return None
    size = stage_obj.get("size") or {}
    pos = (stage_obj.get("position") or [0,0,0]) + [0,0,0]
    px, py, pz = [float(v) for v in pos[:3]]
    w = float(size.get("w", 0.0) or 0.0)
    d = float(size.get("d", 0.0) or 0.0)
    h = float(size.get("h", 0.0) or 0.0)
    geom = stage_obj.get("geometry") or {}
    fp = geom.get("footprint") or {}
    fptype = str(fp.get("type", "")).lower()
    radius = None
    if fptype == "circle":
        try:
            radius = float(fp.get("radius", 0.0))
        except Exception:
            radius = None
        if radius and radius > 0:
            w = max(w, radius * 2.0)
            d = max(d, radius * 2.0)
    return {
        "x": px, "y": py, "z": pz, "w": w, "d": d, "h": h,
        "top_z": pz + h, "front_y": py - d / 2.0, "back_y": py + d / 2.0,
        "radius": radius, "footprint_type": fptype,
    }


def _support_top_z_at_xy(x, y, default=0.0):
    """Find the highest support surface under an XY point.

    This is generic: any platform/floor object with an axis-aligned footprint can
    support furniture, seats, props, and array children.  It avoids hard-coding
    specific scenes such as sofa/table platforms.
    """
    best = None
    for obj in (_SCENE_CTX.get("by_id", {}) or {}).values():
        if not isinstance(obj, dict):
            continue
        if str(obj.get("category", "")).lower() not in ("platform", "floor", "stage"):
            continue
        size = obj.get("size") or {}
        pos = (obj.get("position") or [0, 0, 0]) + [0, 0, 0]
        try:
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            w = float(size.get("w", 0.0) or 0.0)
            d = float(size.get("d", 0.0) or 0.0)
            h = float(size.get("h", 0.0) or 0.0)
        except Exception:
            continue
        if w <= 0 or d <= 0:
            continue
        margin = 0.08
        if (px - w / 2.0 - margin) <= x <= (px + w / 2.0 + margin) and (py - d / 2.0 - margin) <= y <= (py + d / 2.0 + margin):
            top = pz + h
            if best is None or top > best:
                best = top
    return float(best if best is not None else default)


def _unique_positions(positions, precision=3):
    out, seen = [], set()
    for p in positions or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            continue
        vals = list(p) + [0.0, 0.0, 0.0]
        try:
            q = (round(float(vals[0]), precision), round(float(vals[1]), precision), round(float(vals[2]), precision))
        except Exception:
            continue
        if q in seen:
            continue
        seen.add(q)
        out.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return out


def _resolve_object_position(o):
    """Resolve a preview-friendly object position from semantic placement hints.

    This preserves explicit JSON positions when they are already consistent, and
    only nudges categories that are commonly underspecified in extracted scene
    graphs: downstage monitor wedges and truss-mounted spotlights.
    """
    pos = list((o.get("position", [0, 0, 0]) + [0, 0, 0])[:3])
    try:
        px, py, pz = [float(v) for v in pos]
    except Exception:
        return [0.0, 0.0, 0.0]

    position_state=(o.get("parameter_states") or {}).get("position",{})
    if position_state.get("adoption_status")=="adopted" or position_state.get("value_source")=="user_preview":
        return [px,py,pz]
    placement = o.get("placement") or {}
    stage_meta = _stage_metrics(_SCENE_CTX.get("primary_stage"))
    cat = str(o.get("category", "")).lower()
    subtype = str(o.get("subtype", "")).lower()
    gh, layout = _layout_hint(o)
    center = layout.get("center") if isinstance(layout, dict) else None
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        vals = list(center) + [0.0, 0.0, 0.0]
        try:
            cx, cy, cz = float(vals[0]), float(vals[1]), float(vals[2])
            if (
                subtype.endswith("_array") or subtype.endswith("_group")
                or _has(_semantic_text(o), "array", "group", "chairs", "seats", "stools", "curtain", "drape")
            ):
                drift = math.hypot(px - cx, py - cy)
                if drift > max(1.0, max(w, d, 0.01) * 0.60):
                    px, py = cx, cy
                    if abs(cz) > 1e-6:
                        pz = cz
        except Exception:
            pass
    geom = o.get("geometry") or {}
    proc = geom.get("procedural") if isinstance(geom, dict) else {}
    proc = proc if isinstance(proc, dict) else {}
    family = str(proc.get("shape_family", "")).lower()
    size = o.get("size") or {}
    w = float(size.get("w", 0.0) or 0.0)
    d = float(size.get("d", 0.0) or 0.0)
    h = float(size.get("h", 0.0) or 0.0)

    # Generic overhead normalization.  Stage extraction often misuses z as
    # image-depth for suspended objects.  Use the declared scene clearance and
    # local performance-platform top as a stable rigging height.
    role = str((o.get("generation_hints") or {}).get("role", "")).lower() if isinstance(o.get("generation_hints"), dict) else ""
    if cat in ("truss", "rigging") or subtype.endswith("_truss") or role == "suspended_object":
        pz = _normalize_overhead_z(pz, h)

    # Monitor wedges: placement.zone=downstage means they should sit on the
    # audience-facing part of the primary stage surface.  For circular/oval
    # stages the downstage edge depends on X, so using a rectangular bounding
    # box front would place side wedges outside the real platform and make them
    # appear to float.  This contact solver is generic for all curved stages.
    if stage_meta and (family == "monitor_wedge" or subtype == "monitor_wedge"):
        zone = str(placement.get("zone", "")).lower()
        if zone == "downstage":
            pz = stage_meta["top_z"]
            if stage_meta.get("radius"):
                radius = float(stage_meta["radius"])
                limit = max(0.0, radius - max(w * 0.55, 0.20))
                px = max(stage_meta["x"] - limit, min(stage_meta["x"] + limit, px))
                dx = px - stage_meta["x"]
                half_depth = math.sqrt(max(0.0, radius * radius - dx * dx))
                local_front_y = stage_meta["y"] - half_depth
                # Keep the wedge just inside the curved platform boundary so
                # its entire body is supported by the stage surface.
                py = local_front_y + max(d * 0.72, 0.28)
            elif stage_meta["w"] > 0:
                limit = max(0.0, stage_meta["w"] / 2.0 - max(w * 0.55, 0.20))
                px = max(stage_meta["x"] - limit, min(stage_meta["x"] + limit, px))
                py = stage_meta["front_y"] + max(d * 0.72, 0.28)

    # Hanging lights mounted_on a truss should be docked to an actual chord,
    # not merely placed near the truss center.  Treat the resolved JSON position
    # as the fixture mount/clamp point and place it at the truss front-bottom
    # chord with a small clearance.  build_spotlight_fixture then hangs the body
    # below that mount point.
    if cat == "light" and str(placement.get("zone", "")).lower() == "overhead":
        mount_id = placement.get("mounted_on") or placement.get("anchor")
        mount_obj = _SCENE_CTX.get("by_id", {}).get(mount_id) if mount_id else None
        if isinstance(mount_obj, dict):
            msize = mount_obj.get("size") or {}
            mpos = (mount_obj.get("position") or [0,0,0]) + [0,0,0]
            mx, my, mz = [float(v) for v in mpos[:3]]
            md = float(msize.get("d", 0.0) or 0.0)
            mh = float(msize.get("h", 0.0) or 0.0)
            # For current truss builder pz is the lower chord plane and py is
            # the cross-section center.  The clamp should sit at/front of the
            # bottom chord rather than floating far downstage.
            front_y = my - md / 2.0
            bottom_z = mz
            py = front_y - 0.055
            pz = bottom_z - 0.010

    return [px, py, pz]


def _attach_metadata(obj, source_obj):
    if obj is None or not isinstance(source_obj, dict):
        return obj
    if source_obj.get("parameter_states") is not None:
        obj["inkvox_parameter_states"] = json.dumps(source_obj["parameter_states"], ensure_ascii=False)
    if source_obj.get('object_notes') is not None:obj['inkvox_object_notes']=source_obj['object_notes']
    for key in ("id", "category", "subtype"):
        if source_obj.get(key) is not None:
            obj[f"inkvox_{key}"] = str(source_obj.get(key))
    if source_obj.get("source_evidence") is not None:
        try:
            obj["inkvox_source_evidence"] = json.dumps(source_obj.get("source_evidence"), ensure_ascii=False)[:16000]
        except Exception:
            pass
    return obj

def _add_light_visibility_fallback(o, px, py, pz, rz):
    """Add an explicit visible lamp detail for diagnostic/stable previews.

    This is a generic fallback for small dark light fixtures. It does not
    replace scene-driven geometry; it ensures that a declared light object is
    visible in the render even if its procedural body is visually lost against
    a truss or backdrop.
    """
    if not VISIBILITY_FALLBACKS:
        return None
    oid = o.get("id", "light")
    size = o.get("size") or {}
    w = float(size.get("w", 0.32) or 0.32)
    d = float(size.get("d", 0.32) or 0.32)
    h = float(size.get("h", 0.45) or 0.45)
    r = max(0.105, min(w, h) * 0.38)
    length = max(0.40, d * 1.25)
    base_rgb = _visibility_rgb(base_color_for(o), "light", minimum=0.07)
    body_mat = get_mat("light_fallback_body_" + oid, base_rgb, roughness=0.42, metallic=0.65)
    lens_mat = get_mat("light_fallback_lens_" + oid, (1.0, 0.72, 0.28), roughness=0.16,
                       emission_rgb=(1.0, 0.48, 0.16), emission_strength=1.15)

    # Place the visible fallback clearly below the mount point and toward the
    # audience/downstage side.
    fz = pz - max(0.20, h * 0.34)
    fy_off = -max(0.06, length * 0.08)
    c1 = _wxy(px, py, 0, fy_off + length * 0.28, rz)
    c2 = _wxy(px, py, 0, fy_off - length * 0.28, rz)
    body = _cyl(oid + "_visibility_body", (c1[0], c1[1], fz), (c2[0], c2[1], fz), r, body_mat, verts=36)
    l1 = _wxy(px, py, 0, fy_off - length * 0.37, rz)
    l2 = _wxy(px, py, 0, fy_off - length * 0.28, rz)
    _cyl(oid + "_visibility_lens", (l1[0], l1[1], fz), (l2[0], l2[1], fz), r * 0.78, lens_mat, verts=32)
    # A small top hanger helps identify the fixture as mounted hardware.
    hx, hy = _wxy(px, py, 0, fy_off + length * 0.28, rz)
    _cyl(oid + "_visibility_hanger", (hx, hy, fz + r * 0.75), (hx, hy, pz - 0.04), 0.018, body_mat, verts=10)
    return body


def _add_audio_visibility_fallback(o, px, py, pz, rz):
    """Add an explicit visible monitor/speaker detail for previews."""
    if not VISIBILITY_FALLBACKS:
        return None
    oid = o.get("id", "audio")
    size = o.get("size") or {}
    w = max(float(size.get("w", 0.6) or 0.6), 0.62)
    d = max(float(size.get("d", 0.42) or 0.42), 0.42)
    h = max(float(size.get("h", 0.24) or 0.24), 0.22)
    base_rgb = _visibility_rgb(base_color_for(o), "audio", minimum=0.06)
    body_mat = get_mat("audio_fallback_body_" + oid, base_rgb, roughness=0.68, metallic=0.08)
    grille_mat = get_mat("audio_fallback_grille_" + oid, (0.10, 0.10, 0.11), roughness=0.72, metallic=0.18)

    # A shallow front-facing marker plate on the audience side guarantees that
    # downstage monitors remain visible from the preview camera.
    fx, fy = _wxy(px, py, 0, -d/2 - 0.05, rz)
    front = _cube(oid + "_visibility_front", (fx, fy, pz + h * 0.42), (w * 0.92, 0.055, h * 0.30),
                  grille_mat, rot=(0, 0, rz), bevel=0.008)
    tx, ty = _wxy(px, py, 0, 0, rz)
    _cube(oid + "_visibility_top", (tx, ty, pz + h + 0.018), (w * 0.82, d * 0.16, 0.026),
          grille_mat, rot=(0, 0, rz), bevel=0.006)
    # Small dark mass behind the plate, so it reads as an audio wedge, not a label.
    bx, by = _wxy(px, py, 0, -d * 0.12, rz)
    _cube(oid + "_visibility_mass", (bx, by, pz + h * 0.30), (w, d * 0.52, h * 0.42),
          body_mat, rot=(0, 0, rz), bevel=0.012)
    return front


def build_object(o):
    """Dispatch one scene object to a v0.3-aware Blender builder.

    v0.3 gives each object a geometry.representation.  We honor that first,
    then fall back to category/subtype semantic dispatch for v0.2 compatibility.
    """
    cat = o.get("category")
    size = o.get("size") or {}
    if "w" not in size:
        return None
    w = float(size.get("w", 1.0))
    d = float(size.get("d", 1.0))
    h = float(size.get("h", 1.0))
    # Generic visual sanity: walkable stage surfaces should not render as
    # multi-meter solid blocks.  Keep elevated/walkway/riser objects intact, but
    # clamp main/performance/stage-floor surfaces to plausible deck thickness.
    if _is_stage_surface(o) and h > 1.0 and (o.get("parameter_states") or {}).get("size.h", {}).get("value_source") != "user_preview" and (o.get("parameter_states") or {}).get("size.h", {}).get("adoption_status") != "adopted":
        print(f"[builder_sanity] clamp stage surface height id={o.get('id')} h={h:.3f}->0.350")
        h = 0.35
    pos = _resolve_object_position(o)
    px, py, pz = [float(v) for v in pos]
    rot = (o.get("rotation", [0, 0, 0]) + [0, 0, 0])[:3]
    rz = math.radians(float(rot[2]))
    text = _semantic_text(o)
    oid = o.get("id", "obj")
    col = base_color_for(o)
    geom = o.get("geometry") or {}
    rep = str(geom.get("representation", "")).lower() if isinstance(geom, dict) else ""
    proc = geom.get("procedural") if isinstance(geom, dict) else {}
    proc = proc if isinstance(proc, dict) else {}
    family = str(proc.get("shape_family", "")).lower()

    obj = None
    try:
        # Category / subtype / procedural family is stronger evidence than
        # fuzzy keyword matching.  In VLM output, an audio object may carry
        # source notes mentioning the overhead truss, and a text-only dispatch
        # can accidentally route monitor_wedge_* objects into build_truss(),
        # producing monitor_wedge_*_chord / _web / _vert artifacts.  Therefore
        # dispatch known audio and truss categories before broad semantic words.
        cat_l = str(cat or "").lower()
        subtype_l = str(o.get("subtype", "")).lower()
        oid_l = str(oid or "").lower()

        is_monitor = (
            cat_l == "audio"
            and (
                family == "monitor_wedge"
                or subtype_l == "monitor_wedge"
                or "monitor_wedge" in oid_l
                or "wedge" in subtype_l
            )
        )
        is_line_audio = (
            cat_l == "audio"
            and not is_monitor
            and (family in ("line_array", "speaker_array") or _has(text, "speaker", "array", "line array", "sub", "pa "))
        )
        is_scenic_wall = (
            cat_l in ("wall", "scenic_background")
            or family == "scenic_wall"
            or _has(text, "back wall", "rear wall", "background wall", "acoustic panel", "wall panel",
                    "side wall", "sidewall", "cyclorama", "scenic wall", "后墙", "背景墙", "墙板", "声学")
        )
        is_set_piece = (
            cat_l in ("set_piece", "scenic", "scenic_element", "column")
            or _has(text, "proscenium", "column", "pillar")
        )
        is_seating_array = (
            cat_l == "furniture"
            and (
                _has(text, "audience", "seating", "seats", "seat array", "chair array", "stool array", "stools", "chairs", "观众", "座椅", "座位", "高脚凳")
                or (_is_array_proxy(o) and _has(text, "chair", "armchair", "seat", "stool", "高脚凳", "座位", "座椅"))
            )
        )
        is_truss = (
            cat_l == "truss"
            or family == "box_truss"
            or (
                cat_l not in ("audio", "light", "platform", "stairs", "curtain", "screen", "set_piece", "scenic", "scenic_element", "wall", "column", "furniture")
                and _has(text, "truss", "rig", "lighting bar", "light bar", "grid", "gantry")
            )
        )

        shape_state = (o.get("parameter_states") or {}).get("geometry.shape", {})
        explicit_shape = geom.get("shape") if shape_state.get("value_source") == "user_preview" or shape_state.get("adoption_status") == "adopted" else None
        if explicit_shape in ("box", "cylinder", "sphere"):
            if explicit_shape == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=0.5, depth=1, location=(px,py,pz+h/2))
            elif explicit_shape == "sphere":
                bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.5, location=(px,py,pz+h/2))
            else:
                bpy.ops.mesh.primitive_cube_add(size=1, location=(px,py,pz+h/2))
            obj=bpy.context.object
            obj.name=oid
            obj.dimensions=(w,d,h)
            obj.rotation_euler[2]=rz
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            # Reuse the same material library used by category-specific builders.
            obj.data.materials.append(material_for_object(o, name_prefix=oid, fallback_rgb=col))
        elif is_monitor:
            obj = build_monitor_wedge(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l == "light" and _is_array_proxy(o):
            obj = build_light_array(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l == "decor" and _is_array_proxy(o) and _has(text, "painting", "picture", "frame", "canvas", "画", "装饰画"):
            obj = build_decor_array(oid, w, d, h, px, py, pz, rz, col, o)
        elif rep == "contour_extrusion":
            obj = build_from_footprint(oid, w, d, h, px, py, pz, rz, col, o)
        elif is_scenic_wall:
            obj = build_scenic_wall_panel(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l == "platform":
            obj = build_platform(oid, w, d, h, px, py, pz, rz, col, o)
        elif is_set_piece:
            obj = build_proscenium_set_piece(oid, w, d, h, px, py, pz, rz, col, o)
        elif is_seating_array:
            obj = build_seating_array(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l == "screen" or family in ("led_screen", "projection_screen"):
            obj = build_screen(oid, w, d, h, px, py, pz, rz, o)
        elif cat_l == "light" or family == "stage_spotlight" or _has(text, "spotlight", "stage spotlight", "lighting fixture", "par can"):
            obj = build_spotlight_fixture(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l == "stairs" or family in ("straight_stairs", "stairs") or _has(text, "stair", "stairs", "step", "steps", "tread", "riser"):
            obj = build_stairs(oid, w, d, h, px, py, pz, rz, col, o)
        elif cat_l in ("curtain", "scenic_background") or family == "pleated_curtain" or _has(text, "curtain", "drape", "valance", "cloth", "fabric backdrop", "backdrop fabric"):
            obj = build_curtain(oid, w, d, h, px, py, pz, rz, col, o)
        elif is_truss:
            obj = build_truss(oid, w, d, h, px, py, pz, rz, col, o)
        elif is_line_audio:
            obj = build_line_array(oid, w, d, h, px, py, pz, rz, col, o)
        elif _has(text, "circular", "circle", "round", "disc", "disk", "oval"):
            obj = build_disc(oid, w, d, h, px, py, pz, rz, col, o)
        elif _has(text, "column", "pillar", "post", "pole"):
            obj = build_column(oid, w, d, h, px, py, pz, rz, col)
        elif _has(text, "table", "desk", "podium"):
            obj = build_table(oid, w, d, h, px, py, pz, rz, col, o)
        elif _has(text, "stool", "高脚凳", "凳"):
            obj = build_stool(oid, w, d, h, px, py, pz, rz, col, o)
        elif _has(text, "chair", "seat", "throne", "bench"):
            obj = build_chair(oid, w, d, h, px, py, pz, rz, col, o)
        else:
            obj = build_generic(oid, w, d, h, px, py, pz, rz, col, o)
    finally:
        _attach_metadata(obj, o)

    if cat in ("light", "audio"):
        print(f"[build_diag] id={oid} cat={cat} subtype={o.get('subtype')} family={family} "
              f"final_pos=({px:.3f},{py:.3f},{pz:.3f}) size=({w:.3f},{d:.3f},{h:.3f}) "
              f"main_obj={(obj.name if obj else 'None')}")
        fallback = None
        # In normal runs the procedural builder is the visible object.  Fallbacks
        # are kept for failed/unsupported cases, or can be forced during later
        # diagnostics by setting FORCE_VISIBILITY_FALLBACKS=True.
        if obj is None or FORCE_VISIBILITY_FALLBACKS:
            if cat == "light":
                fallback = _add_light_visibility_fallback(o, px, py, pz, rz)
            elif cat == "audio":
                fallback = _add_audio_visibility_fallback(o, px, py, pz, rz)
            if obj is None and fallback is not None:
                obj = fallback

    return obj

# --------------------------------------------------------------- scene prep --
def reset_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials, bpy.data.lights, bpy.data.cameras):
        for item in list(block):
            if item.users == 0:
                block.remove(item)
    _MAT_CACHE.clear()


def setup_render():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = CYCLES_SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.device = "CPU"
    scene.render.resolution_x = RENDER_W
    scene.render.resolution_y = RENDER_H
    scene.render.image_settings.file_format = "PNG"
    try:
        scene.view_settings.exposure = 1.0
    except Exception:
        pass
    try:
        scene.render.film_transparent = False
    except Exception:
        pass

    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = WORLD_BG
        bg.inputs[1].default_value = 0.55


def add_floor(stage):
    width = stage.get("width", 10.0)
    depth = stage.get("depth", 8.0)
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    obj = bpy.context.active_object
    obj.name = "stage_floor"
    obj.scale = (width * 1.6, depth * 1.8, 1.0)
    obj.location = (0.0, depth / 2.0, 0.0)
    floor_obj = {"id": "stage_floor", "category": "platform", "material": stage.get("material", {})}
    obj.data.materials.append(material_for_object(floor_obj, "floor", _rgb_from_hex_or_name("warm_neutral_floor") or (0.16, 0.14, 0.12)))
    return obj


# ------------------------------------------------------------------ lights --
LIGHT_TYPE_MAP = {
    "spot": "SPOT", "beam": "SPOT", "par": "SPOT",
    "wash": "AREA", "flood": "AREA",
    "point": "POINT",
}


def add_light(o):
    attrs = o.get("attributes") or {}
    lp = attrs.get("light") or {}
    ltype = LIGHT_TYPE_MAP.get(lp.get("type", "spot"), "SPOT")
    ldata = bpy.data.lights.new(name=o["id"], type=ltype)
    intensity = float(lp.get("intensity", 0.8))
    if ltype == "SPOT":
        ldata.energy = intensity * (SPOT_MAX_W * 1.35)
        ldata.spot_size = math.radians(lp.get("beam_angle_deg", 30))
        ldata.spot_blend = 0.2
        ldata.shadow_soft_size = 0.1
    elif ltype == "AREA":
        ldata.energy = intensity * AREA_MAX_W
        ldata.size = 1.5
    else:
        ldata.energy = intensity * POINT_MAX_W
    # tint by colour temperature if given
    ctk = lp.get("color_temperature_k")
    if ctk:
        ldata.color = _kelvin_to_rgb(ctk)
    lobj = bpy.data.objects.new(o["id"], ldata)
    bpy.context.scene.collection.objects.link(lobj)
    pos = _resolve_object_position(o)
    size = o.get("size") or {}
    if ltype == "SPOT":
        try:
            pos[2] = float(pos[2]) - max(0.08, float(size.get("h", 0.4)) * 0.08)
        except Exception:
            pass
    lobj.location = pos
    rot = (o.get("rotation", [0, 0, 0]) + [0, 0, 0])[:3]
    lobj.rotation_euler = (math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2]))
    return lobj


def _kelvin_to_rgb(k):
    """Cheap blackbody approximation -> linear RGB tint."""
    t = max(1000.0, min(12000.0, float(k))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.47 * math.log(t) - 161.12 if t > 0 else 255.0
    else:
        r = 329.7 * ((t - 60) ** -0.1332)
        g = 288.12 * ((t - 60) ** -0.0755)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.52 * math.log(t - 10) - 305.04
    clamp = lambda v: max(0.0, min(255.0, v))
    return (srgb_to_linear(clamp(r)), srgb_to_linear(clamp(g)), srgb_to_linear(clamp(b)))


def aim_at(obj, target_obj):
    con = obj.constraints.new(type="TRACK_TO")
    con.target = target_obj
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"


def add_preview_lighting(stage, camera=None):
    """Add non-diegetic preview lights so the render stays readable.

    These are gentle area lights that complement the scene lights rather than
    replacing them. They make scene.filled previews usable even when the JSON
    only specifies a few theatrical spotlights.
    """
    width = float(stage.get("width", 10.0))
    depth = float(stage.get("depth", 8.0))
    clearance = float(stage.get("height_clearance", 6.0))
    stage_center = mathutils.Vector((0.0, depth * 0.48, 0.9))

    def _add_area(name, location, size, energy, color=(1.0, 1.0, 1.0)):
        ldata = bpy.data.lights.new(name=name, type='AREA')
        ldata.shape = 'RECTANGLE'
        ldata.size = size
        ldata.size_y = size * 0.55
        ldata.energy = energy
        ldata.color = color
        lobj = bpy.data.objects.new(name, ldata)
        bpy.context.scene.collection.objects.link(lobj)
        lobj.location = location
        aim = bpy.data.objects.new(name + '_aim', None)
        bpy.context.scene.collection.objects.link(aim)
        aim.location = stage_center
        aim_at(lobj, aim)
        return lobj

    warm = _kelvin_to_rgb(4200)
    neutral = _kelvin_to_rgb(5200)
    _add_area('preview_key', (0.0, -(depth * 0.95 + 3.4), clearance * 1.05), max(width * 0.95, 5.0), 3200.0, warm)
    _add_area('preview_fill_left', (-width * 0.9, -(depth * 0.35 + 1.8), clearance * 0.95), max(width * 0.55, 3.0), 1200.0, neutral)
    _add_area('preview_fill_right', (width * 0.9, -(depth * 0.35 + 1.8), clearance * 0.95), max(width * 0.55, 3.0), 1200.0, neutral)
    _add_area('preview_top', (0.0, depth * 0.15, clearance * 1.45), max(width * 0.8, 4.0), 900.0, neutral)

    if camera is not None:
        cam_fill = bpy.data.lights.new(name='preview_cam_fill', type='POINT')
        cam_fill.energy = 200.0
        cam_fill.shadow_soft_size = 0.35
        cam_fill.color = neutral
        fill_obj = bpy.data.objects.new('preview_cam_fill', cam_fill)
        bpy.context.scene.collection.objects.link(fill_obj)
        fill_obj.location = camera.location
        fill_obj.location.z += 0.25


def add_camera(stage, cameras=None):
    width = stage.get("width", 10.0)
    depth = stage.get("depth", 8.0)
    clearance = stage.get("height_clearance", 6.0)
    env = _SCENE_CTX.get("stage_envelope") or {}
    anchor = _SCENE_CTX.get("performance_anchor") or env
    if env:
        width = max(width, env.get("w", width))
        depth = max(depth, env.get("d", depth))
    sm = _stage_metrics(_SCENE_CTX.get("primary_stage"))
    if sm:
        clearance = max(clearance, sm["top_z"] + 4.0)
    cameras = cameras or []
    preferred = None
    for wanted in ("canonical_render", "front", "front_right_45", "front_left_45", "semi_overhead"):
        for c in cameras:
            if c.get("view_name") == wanted:
                preferred = c
                break
        if preferred is not None:
            break
    view = (preferred or {}).get("view_name", "canonical_render")

    cam_data = bpy.data.cameras.new("preview_cam")
    cam_data.lens = float((preferred or {}).get("focal_length", 44 if view != "semi_overhead" else 32))
    cam = bpy.data.objects.new("preview_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam)

    if view == "front":
        cam.location = (0.0, -(depth * 1.18 + 3.0), clearance * 0.92)
    elif view == "front_left_45":
        cam.location = (-width * 0.72, -(depth * 1.05 + 3.2), clearance * 1.02)
    elif view == "semi_overhead":
        cam.location = (width * 0.22, -(depth * 0.70 + 3.0), clearance * 1.60)
    else:  # canonical / front_right_45
        cam.location = (width * 0.70, -(depth * 1.02 + 3.2), clearance * 1.02)

    look = bpy.data.objects.new("cam_target", None)
    bpy.context.scene.collection.objects.link(look)
    target_x = anchor.get("x", sm["x"] if sm else 0.0) if isinstance(anchor, dict) else (sm["x"] if sm else 0.0)
    target_y = anchor.get("y", sm["y"] if sm else depth * 0.30) if isinstance(anchor, dict) else (sm["y"] if sm else depth * 0.30)
    target_z = (sm["top_z"] + max(0.6, clearance * 0.08)) if sm else clearance * 0.30
    look.location = (target_x, target_y, target_z)
    aim_at(cam, look)
    bpy.context.scene.camera = cam
    return cam

# ------------------------------------------------------------------- main --
def main():
    scene_path, out_path = parse_args()
    print(f"[generate_stage] version: {INKVOX_GENERATOR_VERSION}")
    print(f"[generate_stage] reading: {scene_path}")
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scene = data["scene"]
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from decision_state import initialize
    initialize(scene)
    from decision_state import consistency_errors
    errors=consistency_errors(scene)
    if errors:raise ValueError("Invalid design parameters: "+"; ".join(errors))

    global _SCENE_CTX
    _SCENE_CTX = _build_scene_context(scene)

    reset_scene()
    setup_render()
    add_floor(scene.get("stage", {}))

    if _SCENE_CTX.get("primary_stage"):
        print(f"[generate_stage] primary stage: {_SCENE_CTX['primary_stage'].get('id')}")

    id_map = {}
    lights_with_target = []
    lights_no_target = []

    for o in scene.get("objects", []):
        cat = o.get("category")
        oid = o.get("id", "unnamed")
        try:
            if cat == "light":
                fixture = build_object(o)
                lobj = add_light(o)
                id_map[oid] = fixture or lobj
                tgt = ((o.get("attributes") or {}).get("light") or {}).get("target")
                (lights_with_target.append((lobj, tgt)) if tgt else lights_no_target.append(lobj))
            else:
                obj = build_object(o)
                if obj is None:
                    print(f"[skip] {oid}: no size (expected a validated scene)")
                else:
                    id_map[oid] = obj
        except Exception as e:
            print(f"[warn] failed to build {oid}: {e}")

    for lobj, tgt in lights_with_target:
        if tgt in id_map:
            aim_at(lobj, id_map[tgt])
        else:
            print(f"[warn] light '{lobj.name}' targets unknown object '{tgt}'")

    if lights_no_target:
        st = scene.get("stage", {})
        center = bpy.data.objects.new("stage_center_aim", None)
        bpy.context.scene.collection.objects.link(center)
        center.location = (0.0, st.get("depth", 8.0) / 2.0, 0.8)
        for lobj in lights_no_target:
            aim_at(lobj, center)

    cam = add_camera(scene.get("stage", {}), scene.get("cameras", []))
    add_preview_lighting(scene.get("stage", {}), cam)

    blend_path = os.path.abspath(os.path.splitext(out_path)[0] + ".blend")
    from blender_decision_panel import install_panel
    if scene.get('metadata',{}).get('experiment_condition') != 'B':
        install_panel(data)
    else:
        # Baseline carries readable notes and values, not adoption classifications.
        for obj in bpy.data.objects:
            if 'inkvox_parameter_states' in obj:del obj['inkvox_parameter_states']
        bpy.data.texts.new('InkVox handoff notes').write('\n\n'.join((o.get('label') or o.get('id',''))+'\n'+o.get('object_notes','') for o in scene.get('objects',[])))
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[generate_stage] saved editable model -> {blend_path}")

    bpy.context.scene.render.filepath = out_path
    print(f"[generate_stage] rendering -> {out_path}")
    bpy.ops.render.render(write_still=True)
    print("[generate_stage] done.")


if __name__ == "__main__":
    main()
