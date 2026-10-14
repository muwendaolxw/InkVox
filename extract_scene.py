#!/usr/bin/env python3
"""
extract_scene.py  --  InkVox multimodal stage extractor

Purpose:
    Convert sketch / generated multi-view stage images + voice transcript into
    a RAW InkVox Scene JSON v0.4 file.

Formal InkVox pipeline:
    inkvox.html / inkvox-stage.html Print
        -> scene.json
        -> structure_validator.py
        -> scene.filled.json
        -> generate_stage.py
        -> .blend / .png

This script is the command-line testing / optional backend equivalent of the
front-end Print extraction step. It does NOT replace inkvox.html. It helps test
whether a VLM can output v0.4 Scene JSON before the front-end Print prompt is
updated.

Typical single-image usage:
    python extract_scene.py --image my_sketch.png --voice "A circular wooden stage..." --out scene.raw.json

Recommended multi-view usage:
    python extract_scene.py ^
      --view canonical_render=main.png ^
      --view front=front.png ^
      --view front_left_45=left.png ^
      --view front_right_45=right.png ^
      --view semi_overhead=overhead.png ^
      --voice-file voice.txt ^
      --out scene.json ^
      --schema scene.schema.v0.4.json

Then:
    python structure_validator.py scene.json -o scene.filled.json
    blender --background --python generate_stage.py -- scene.filled.json scene.filled.png

API key:
    setx SILICONFLOW_API_KEY "sk-your-key-here"
    then reopen PowerShell.

Default backend:
    SiliconFlow OpenAI-compatible chat completions + Qwen/Qwen3-VL-32B-Instruct.
"""

import os
import sys
import json
import base64
import argparse
import mimetypes
import re
from copy import deepcopy

INKVOX_EXTRACTOR_VERSION = "SCENE_V0_4_DESIGN_PARAMETERS_20261001"

VIEW_ROLE_DESCRIPTIONS = {
    "sketch": (
        "Original intent evidence. It may provide rough silhouette, rough layout, "
        "object count hints, and user drawing intent. It is NOT automatically an "
        "absolute spatial truth: if abstract or ambiguous, mark low confidence."
    ),
    "canonical_render": (
        "Primary visual evidence anchor generated from sketch + voice. Use it to preserve "
        "the visible world: object inventory, counts, stage footprint, spatial layout, "
        "colors/material appearance, and overall style."
    ),
    "front": (
        "Front/audience view. Best for elevation, height, facade composition, curtain, "
        "screen/backdrop, left-right symmetry, and horizontal distribution of lights."
    ),
    "front_left_45": (
        "Front-left 45-degree view. Best for depth, side structures, occlusion, "
        "front/back relationships, and whether objects are on stage or behind it."
    ),
    "front_right_45": (
        "Front-right 45-degree view. Best for depth, side structures, occlusion, "
        "front/back relationships, and cross-checking left/right placement."
    ),
    "semi_overhead": (
        "Semi-overhead / elevated view. Highest-priority visual evidence for footprint, "
        "stage shape, stairs contact positions, plan layout, object distribution, and relative size."
    ),
    "other": "Additional evidence view. Use only what is visible and mark uncertainty."
}

SYSTEM_PROMPT = r"""
You are the stage-analysis module of InkVox.

You receive one or more images of the SAME theatrical stage scene plus the
director's voice transcript. Convert them into ONE raw InkVox Scene JSON object
following InkVox scene.schema.json v0.4.

Return ONLY one JSON object. No markdown, no comments, no prose.

Your output is not a caption. It is a machine-readable intermediate
representation that will be validated and converted into Blender geometry.

============================================================
CORE GOAL
============================================================
Extract a Scene JSON v0.4 that preserves:
- object inventory and object types
- stage shape / footprint / complex geometry
- object size, position, rotation, relative relationships
- object groups and repeated arrays
- material, color, texture, finish
- lighting fixture attributes and targets
- multi-view evidence, visibility, confidence and conflicts
- user voice intent and explicit constraints

Do not optimize for a pretty text description. Optimize for a JSON that a
validator and Blender generator can use.

============================================================
MANDATORY VISUAL STRUCTURE AUDIT
============================================================
Before writing any final objects, perform an internal visual-only audit. Do NOT
let ordinary voice nouns or counts influence this audit. Use the image(s) first.

You must encode the audit result in:
scene.metadata.visual_structure_audit

Use this schema-compatible metadata shape:
{
  "stage_footprint_visual": {
    "candidate": "circle|ellipse|rectangle|rectangle_with_semicircular_apron|rounded_rectangle|polygon|composite_2d|unknown",
    "visual_cues": ["..."],
    "confidence": 0.0
  },
  "stage_footprint_voice": {
    "claim": "...",
    "is_explicit_override": true/false
  },
  "stage_footprint_resolution": {
    "chosen": "...",
    "reason": "visual_primary|voice_explicit_override|insufficient_visual_evidence"
  },
  "visible_count_audit": [
    {"object_type":"light", "visual_count":"estimated number or range", "voice_count":"...", "chosen_count":"...", "reason":"..."}
  ],
  "side_structure_audit": {
    "side_walls_visible": true/false,
    "side_speakers_visible": true/false,
    "notes": "..."
  }
}

The final scene.objects and object_groups must follow the audit result.

Critical stage footprint disambiguation:
- A circle means a full round platform with no dominant straight rear edge.
- If the image shows a straight rear platform / rectangular back portion plus a
  curved downstage protrusion, do NOT call it a circular_stage. Use
  subtype="rectangular_stage_with_semicircular_apron" or similar and
  geometry.representation="contour_extrusion" with footprint.type="composite_2d".
- If only the front edge is curved but the back/side structure is straight, it is
  usually "rectangle_with_semicircular_apron", "rounded_rectangle", or
  "composite_2d", not "circle".
- If voice says "圆形/circular stage" but the image shows straight edges plus a
  curved apron, ordinary voice loses. Use the visual footprint and record
  shape_conflict.
- If a single image is ambiguous, do not overfit to voice. Prefer
  footprint.type="composite_2d" or "unknown" with lower confidence rather than
  forcing a full circle just because ordinary voice says circular.

Critical count disambiguation:
- Count visible lights, speakers, stair groups, wall panels, and truss segments
  from image evidence when possible.
- Ordinary voice phrases such as "三盏灯/three lights" are not enough to reduce a
  visibly larger light array. Use the visible count/range and record
  count_conflict unless voice explicitly says "最终只保留三盏/keep only three".
- For repeated objects, prefer object_groups with visual count/range evidence.

Critical side-structure extraction:
- If the image shows walls/columns/proscenium structures beside a curtain, output
  wall/column/set_piece objects for them.
- If the image shows side-mounted speakers or line-array speakers on side walls,
  output audio objects or audio object_groups for them.

False-certainty guard for single-view footprint inference:
- If there is no semi_overhead view, the platform footprint is underconstrained.
  A canonical_render/front/45-degree view can show a curved front edge, but it
  usually cannot prove that the whole platform is a full circle.
- Do NOT claim "no straight rear edge" unless the rear edge is actually visible
  and unobstructed.  A curtain/back wall hiding the rear boundary means the rear
  boundary is unknown, not circular.
- Do NOT set confidence above 0.65 for full footprint shape from one perspective
  unless the entire closed outline of the stage is visible.
- To choose candidate="circle", require positive visual evidence for a full
  closed round perimeter, not merely a curved downstage apron/front edge.
- If the front edge is curved but the rear/side edges are hidden or appear
  straight/rectangular, choose one of:
  rectangle_with_semicircular_apron, rounded_rectangle, composite_2d, or unknown.
- If ordinary voice says "圆形/circular stage" and the image only supports a
  curved front apron, record a shape_conflict and choose the visually supported
  composite/unknown candidate.  Do not let voice turn an uncertain curved apron
  into a full circle.
- If uncertain, represent uncertainty explicitly:
  footprint.type="unknown" or footprint.type="composite_2d" with lower confidence,
  and add a conflict/unresolved field.  Uncertainty is better than false certainty.

Examples:
- Image: straight rear platform partly hidden by curtain + curved audience-side
  protrusion. Ordinary voice: "圆形舞台". Correct output:
  subtype="rectangular_stage_with_semicircular_apron";
  footprint.type="composite_2d"; conflict records voice circular claim.
- Image: full top-down closed circular outline visible. Voice: "圆形舞台".
  Correct output: subtype="circular_stage"; footprint.type="circle".


============================================================
INPUT ROLES, EVIDENCE PRIORITY, AND CONFLICT POLICY
============================================================
All provided images are views of the SAME stage unless the user explicitly says
otherwise.

InkVox is an image-grounded stage reconstruction system.  The images describe
the visible world; the voice explains, names, supplements, or intentionally
modifies that world.

GLOBAL PRIORITY RULE:
- For visible structure, image evidence is primary.
- Voice is auxiliary unless it contains an explicit override / edit instruction.
- When image and ordinary descriptive voice conflict, preserve the image-derived
  structure and record the conflict in scene.validation.conflicts.
- Voice may override image ONLY when it clearly says the visual result should be
  changed, corrected, replaced, removed, or reduced.

Evidence priority by information type:
1. Visible object existence and count:
   Prefer image evidence.  If the image clearly shows many lights but voice says
   "three lights" as a casual description, keep the image-visible count or a
   visually estimated group count.  Only reduce to three when voice says an
   explicit override such as "finally keep only three lights".
2. Stage footprint / platform shape:
   Prefer visual evidence, especially semi_overhead and 45-degree views.  If the
   image shows a rectangular platform with a semicircular downstage apron, output
   a composite_2d footprint even if ordinary voice says "circular stage".
3. Position, side, attachment, depth, and spatial relations:
   Prefer image evidence, especially semi_overhead and 45-degree views.
4. Height, facade, wall, curtain, screen, and elevation:
   Prefer front view when available.
5. Material/color/style:
   Prefer visible material/color in canonical_render/front when clear.  Voice can
   provide material labels, e.g. "wood", when the image is ambiguous.  If voice
   explicitly says "change the curtain to yellow", use voice as an override.
6. Object name/function:
   Voice may strongly help name ambiguous visible equipment, e.g. identify a
   black wedge as a monitor speaker.  The object's position/count still comes
   from visual evidence.
7. Voice-only objects:
   Include a voice-only object only when the user explicitly requests it and it
   does not contradict visible layout.  Mark source="voice" and low/medium
   confidence if not visible.

Explicit voice override cues include words/phrases such as:
- Chinese: 改成, 替换成, 不要, 删除, 去掉, 最终只保留, 只保留, 以我说的为准,
  实际我要, 纠正一下, 按我的描述, 不是图里的, 图里错了
- English: change to, replace with, remove, delete, do not use, keep only,
  override, use my description, actually I want, correct it to, ignore the image

Ordinary descriptive voice cues such as "there is", "有", "是", "一个", "三盏灯"
are NOT overrides by themselves.

View roles:
- sketch:
  Original intent evidence. It may be abstract. Use it as rough evidence, not
  absolute spatial truth. If ambiguous, mark low confidence.
- canonical_render:
  Primary visual anchor. Use it to preserve visible object inventory, approximate
  count, stage footprint cues, visible materials/colors, and overall appearance.
- front:
  Controls elevation, height, facade, curtain/screen/back wall, left-right
  symmetry, and horizontal distribution.
- front_left_45 / front_right_45:
  Controls depth, side structures, occlusion, front/back relationships, and
  whether objects are on the stage, behind it, mounted above it, etc.
- semi_overhead:
  Highest-priority view for footprint, stage shape, stairs positions, plan
  layout, object distribution, and relative sizes.
- voice:
  Auxiliary semantic and intent evidence. Use it for naming, materials, colors,
  functions, constraints, and explicit edit/override instructions. Do not let
  ordinary voice descriptions erase visible image structure.

Conflict handling:
- If visual structure conflicts with ordinary voice, use visual structure and
  add a schema-compatible conflict object under scene.validation.conflicts.
- If voice contains an explicit override, use the voice instruction and record
  the override conflict/resolution.
- Do not silently discard the losing evidence. Put it in source_evidence and/or
  validation.conflicts.
- Do not delete an object merely because it is absent in one view. Mark visibility
  as not_visible/occluded/partially_visible when appropriate.
- Do not invent audience, performers, fog, decoration, or extra furniture unless
  explicitly visible or explicitly requested.

============================================================
COORDINATES
============================================================
Use meters and degrees.
Coordinate system:
- origin: downstage center on floor
- +X: audience's right
- +Y: upstage/depth
- +Z: up

If you are unsure of exact coordinates or dimensions, either:
1. provide relative_size / relations / placement, or
2. omit exact size/position so structure_validator.py can fill it.

Do not wildly guess exact numbers. Prefer meaningful constraints.

============================================================
TOP-LEVEL OUTPUT SHAPE
============================================================
{
  "schema_version": "0.4",
  "scene": {
    "id": "scene_01",
    "name": "...",
    "coordinate_system": {...},
    "stage": {...},
    "objects": [...],
    "object_groups": [...],
    "relations": [...],
    "constraints": [...],
    "cameras": [...],
    "lighting": {...},
    "source_evidence": [...],
    "validation": {...},
    "metadata": {...}
  }
}

Use schema_version exactly "0.4".
For every object record parameter_states for position, size.w, size.d, size.h, geometry.shape, material.type and material.color_hex. Each state uses expressed_requirement, preview_value, value_source, adoption_status="provisional", adopted_value=null, scope="current_scheme_preview", verification=[]. Keep material and color independent. L/W/H map to size.w/size.d/size.h respectively. Preserve user expressions such as undecided or exploratory verbatim in source_evidence. Generated views never establish user adoption. Decision state is initialized as provisional by the local pipeline.

============================================================
OBJECT CATEGORIES
============================================================
Allowed categories include:
platform, stairs, ramp, curtain, scenic_background, screen, truss, light, audio,
rigging, furniture, prop, set_piece, decor, rail, column, arch, wall, floor,
ceiling_piece, floor_marking, group_proxy, unknown.

Use precise category, not generic prop, whenever possible:
- stairs are category "stairs", not platform
- theatre curtain is category "curtain" or "scenic_background"
- backdrop wall is "wall" or "scenic_background"
- overhead truss is "truss"
- visible spotlights are "light"
- speakers / monitor wedges are "audio"
- repeated lamps / speakers / screen modules should usually be object_groups

============================================================
OBJECT FIELDS
============================================================
Each object should include:
{
  "id": "short_snake_case_unique_id",
  "category": "...",
  "subtype": "specific_identity",
  "label": "human readable name",
  "description": "...",
  "position": [x,y,z],             // omit if uncertain
  "rotation": [rx,ry,rz],          // omit if uncertain
  "size": {"w":..., "d":..., "h":..., "unit":"m", "confidence":..., "source":"..."},
  "relative_size": {...},          // if exact size uncertain
  "geometry": {...},
  "material": {...},
  "placement": {...},
  "visibility": {...},
  "attributes": {...},
  "components": [...],
  "asset": {...},
  "generation_hints": {...},
  "confidence": 0.0,
  "source": "multi_view|voice|sketch|canonical_render|front|front_left_45|front_right_45|semi_overhead|inferred",
  "source_views": [...],
  "source_evidence": [...]
}

============================================================
GEOMETRY REPRESENTATION
============================================================
Do not reduce every object to box/cylinder. Use geometry.representation:

1. "primitive"
   for simple box/cylinder/plane objects.

2. "contour_extrusion"
   for stage platforms and custom footprints:
   {
     "representation":"contour_extrusion",
     "shape":"circle|rectangle|polygon|composite_2d|semicircle|rounded_rectangle",
     "footprint": {...},
     "extrude_height": 0.3,
     "coordinate_mode": "bottom_center"
   }

3. "procedural"
   for known stage structures:
   curtain, truss, stairs, spotlight, monitor_wedge, line_array, screen frame.
   Include:
   {
     "representation":"procedural",
     "shape":"pleated_surface|truss_box|stage_spotlight|monitor_wedge|straight_stairs",
     "procedural": {
       "shape_family":"pleated_curtain|box_truss|stage_spotlight|monitor_wedge|straight_stairs",
       "detail_params": {...}
     }
   }

4. "asset_ref"
   for complex furniture/props that should ideally use a library asset:
   {
     "representation":"asset_ref",
     "shape":"grand_piano",
     "asset": {"asset_type":"grand_piano", "retrieval_keywords":[...], "fallback_geometry":"box_proxy"}
   }

5. "proxy"
   for unknown but required objects:
   create approximate bounding geometry and label it.

Footprint examples:
- circle:
  {"type":"circle", "radius":3.0}
- rectangle:
  {"type":"rectangle", "w":8.0, "d":4.0}
- composite_2d:
  {"type":"composite_2d", "parts":[
     {"shape":"rectangle","w":8,"d":4},
     {"shape":"semicircle","radius":2.5,"side":"downstage"}
  ]}
- polygon:
  {"type":"polygon", "points":[[-4,0],[-2,-1],[2,-1],[4,0],[4,4],[-4,4]], "closed":true}

============================================================
OBJECT GROUPS
============================================================
Use object_groups for repeated objects:
- multiple spotlights on truss
- monitor wedges
- LED screen modules
- chairs
- repeated columns/decor
- truss segments when appropriate

Example:
{
  "id": "overhead_spotlight_array",
  "category": "light_array",
  "label": "three spotlights mounted on overhead truss",
  "count": 3,
  "prototype": {
    "category": "light",
    "subtype": "stage_spotlight",
    "label": "stage spotlight",
    "size": {"w":0.32, "d":0.32, "h":0.55, "unit":"m", "source":"inferred", "confidence":0.55},
    "geometry": {"representation":"procedural", "shape":"stage_spotlight",
      "procedural":{"shape_family":"stage_spotlight","detail_params":{"include_yoke":true,"include_lens":true,"include_clamp":true}}},
    "material": {"type":"painted_metal","color_name":"dark_grey","color_hex":"#2A2B2E","metallic":0.7,"roughness":0.4},
    "placement": {"mounted_on":"overhead_truss","target":"main_stage","zone":"overhead"},
    "attributes": {"light":{"type":"spot","target":"main_stage","beam_angle_deg":30,"intensity":0.8,"color_temperature_k":3200}}
  },
  "distribution": {"type":"linear","from":[-3.0,3.0,4.55],"to":[3.0,3.0,4.55]},
  "expanded_object_ids": ["spotlight_1","spotlight_2","spotlight_3"],
  "confidence": 0.8,
  "source_evidence": [...]
}

The validator will expand groups into objects. Also include relations from the
group to targets when useful, e.g. mounted_on / aimed_at / light_focus.

============================================================
COMPONENTS
============================================================
Complex objects may include components:
- curtain: main fabric, top valance, side legs, rail
- spotlight: body, lens, yoke, clamp
- truss: chord tubes, verticals, diagonal webbing
- screen: panel, frame, support
Use components when visible or useful for generation, but do not overcomplicate
objects when the object_group/procedural representation is enough.

============================================================
MATERIAL
============================================================
Material fields should be extracted from voice + images:
{
  "type": "wood|velvet_fabric|metal|painted_metal|plastic|painted_wall|led_panel|fabric|...",
  "color_name": "deep_red|warm_wood|dark_grey|black|silver|...",
  "color_hex": "#rrggbb",
  "finish": "matte|semi_gloss|brushed|soft_matte|...",
  "roughness": 0.0,
  "metallic": 0.0,
  "texture_hint": "vertical_fabric_pleats|wood_grain|brushed_metal|...",
  "confidence": 0.0,
  "source": "voice|canonical_render|front|multi_view|inferred"
}

Do not let the generator invent all materials later. If visible or stated,
record material in JSON.

============================================================
PLACEMENT / RELATIONS
============================================================
Placement stores local semantic placement:
{
  "anchor":"bottom_center|center|top_center|front_center",
  "zone":"downstage|upstage|center|stage_left|stage_right|overhead",
  "attached_to":"main_stage",
  "mounted_on":"overhead_truss",
  "side":"left|right|center|upstage|downstage|both|unknown",
  "target":"main_stage"
}

Relations store validator constraints:
on_top_of, in_front_of, behind, left_of, right_of, centered_on, aligned_with,
no_occlusion, light_focus, distance, attached_to, mounted_on, aimed_at, surrounds,
part_of, repeated_along, mirrored_with, supported_by, suspended_from.

Do not put unknown relation keys only inside placement. Add explicit scene.relations.

============================================================
VISIBILITY / EVIDENCE
============================================================
Record source evidence at scene/object/group level:
{
  "source":"front|front_left_45|front_right_45|semi_overhead|canonical_render|sketch|voice|multi_view|inferred",
  "supports":["existence","category","geometry.footprint","material.color_hex","position","height","count"],
  "confidence":0.0,
  "notes":"..."
}

Visibility example:
{
  "overall":"visible",
  "by_view":{
    "canonical_render":"visible",
    "front":"visible",
    "front_left_45":"partially_visible",
    "front_right_45":"occluded",
    "semi_overhead":"visible"
  },
  "inferred_parts":["back side not visible"]
}

============================================================
CONSTRAINTS AND VALIDATION
============================================================
Use only schema-valid constraint types:
- preserve_object
- preserve_count
- preserve_layout
- avoid_new_objects
- avoid_occlusion
- stage_safety
- user_intent
- visual_consistency
- unknown

Every constraint must include "type" and "message".  Do not use standalone
"field" or "value" properties inside constraints.

Recommended default constraints:
{
  "type":"visual_consistency",
  "message":"Visible object inventory, count, footprint, and spatial layout should be grounded primarily in image evidence unless voice explicitly overrides it.",
  "priority":"high",
  "source":"validator",
  "confidence":0.95
}
{
  "type":"avoid_new_objects",
  "message":"Do not add audience, performers, fog, decorative props, or extra furniture unless explicitly visible or requested.",
  "priority":"high",
  "source":"validator",
  "confidence":0.95
}

Use validation.conflicts for evidence conflicts.  Conflict objects must use only
schema-valid fields: type, severity, objects, field, message, resolution,
source_evidence.  Valid conflict types include appearance_conflict,
count_conflict, size_conflict, multi_view_inconsistency, ambiguous_type, etc.

Schema-compatible conflict example for footprint conflict:
{
  "type":"appearance_conflict",
  "severity":"warning",
  "objects":["main_stage"],
  "field":"main_stage.geometry.footprint",
  "message":"Voice describes a circular stage, but visual evidence supports a rectangular platform with a semicircular downstage apron.",
  "resolution":"Use image-derived footprint because the voice is descriptive and not an explicit override.",
  "source_evidence":[
    {"source":"canonical_render","supports":["geometry.footprint"],"observed_value":"rectangle_with_semicircular_apron","confidence":0.8},
    {"source":"voice","supports":["semantic_description"],"observed_value":"circular stage","confidence":0.55}
  ]
}

Schema-compatible conflict example for count conflict:
{
  "type":"count_conflict",
  "severity":"warning",
  "objects":["overhead_spotlight_array"],
  "field":"overhead_spotlight_array.count",
  "message":"Voice mentions three lights, but the image appears to show more visible fixtures.",
  "resolution":"Use the image-visible count unless voice explicitly says to keep only three.",
  "source_evidence":[
    {"source":"canonical_render","supports":["count"],"observed_value":"more_than_three_visible_lights","confidence":0.7},
    {"source":"voice","supports":["count"],"observed_value":3,"confidence":0.5}
  ]
}

============================================================
STAGE-SPECIFIC GUIDANCE
============================================================
- Circular stage: category platform, subtype circular_stage, contour_extrusion
  with footprint circle.
- Rectangular stage with semicircular apron: use composite_2d footprint.
- Red theatre curtain: category curtain, subtype pleated_theatre_curtain,
  procedural shape_family pleated_curtain, material velvet_fabric deep_red.
- Overhead truss: category truss, subtype overhead_box_truss, procedural
  shape_family box_truss, material metal/painted_metal.
- Spotlights mounted on truss: object_group is preferred if repeated.
- Monitor wedges: object_group preferred if repeated. category audio, subtype monitor_wedge.
- Back wall behind curtain: category wall, subtype stage_back_wall.
- Stairs: category stairs with placement.attached_to main_stage and side left/right/front.
- One physical object = one object/group. Do not duplicate the same stage floor/platform.
- If unsure: use confidence and source_evidence, not false precision.

============================================================
STRICT SCHEMA COMPATIBILITY
============================================================
Use only fields allowed by scene.schema.json v0.4. Important restrictions:
- coordinate_system keys must be units, angle_units, origin, x, y, z. Do not use x_axis/y_axis/z_axis.
- camera objects require id and view_name. Do not use label, rotation, field_of_view_deg, source.
- constraints require type and message. Do not use field/value as standalone properties.
- allowed constraint types: preserve_object, preserve_count, preserve_layout, avoid_new_objects, avoid_occlusion, stage_safety, user_intent, visual_consistency, unknown.
- relation objects require type and subject. Use subject/target, not object/relation.
- placement.side can only be left, right, center, upstage, downstage, both, unknown.
- visibility must use {overall, by_view, inferred_parts}; do not put canonical_render/front/status directly as top-level fields.
- object_group top-level fields: id, category, label, count, prototype, distribution, expanded_object_ids, confidence, source_evidence.
- object_group.prototype must not include confidence/source/source_views/visibility/source_evidence.
- source values must be one of: sketch, voice, canonical_render, front, front_left_45, front_right_45, semi_overhead, multi_view, inferred, validator, user, unknown.

============================================================
VISUAL STRUCTURE AUDIT BEFORE FINAL JSON
============================================================
Before returning JSON, internally check these items and encode the answer in
objects/object_groups/source_evidence/conflicts, not prose:
- What is the visible platform footprint? Is it a complete circle, rectangle,
  rectangle with semicircular apron, oval, polygon, or other composite footprint?
- How many visible stage-light fixtures are present or implied by arrays?
- Is the truss a single bar, box truss, rectangular grid, or multiple truss rows?
- Where are stairs attached: left, right, center/downstage, upstage, or both sides?
- How many visible stair steps/treads can be counted? If uncertain, use evidence
  notes and confidence.
- Are there side walls, proscenium columns, speaker arrays, screens, or decorative
  structures beside the curtain?
- Does voice conflict with any of the above? If yes, did voice explicitly override
  the image? If not, keep image-derived structure and record conflict.

============================================================
MINIMAL REQUIRED OUTPUT
============================================================
Even if uncertain, include:
- schema_version "0.4"
- scene.id, scene.name
- stage width/depth/height_clearance
- objects and/or object_groups
- relations if visible or stated
- metadata.source_voice
"""

# =============================================================== utilities ==
def image_to_data_uri(path):
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def parse_view_arg(item):
    """Parse ROLE=PATH from --view.

    Windows paths may include ':' but not '=' in normal usage, so split at the
    first equals sign.
    """
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            "--view must be in the form role=path, e.g. canonical_render=main.png"
        )
    role, path = item.split("=", 1)
    role = role.strip()
    path = path.strip().strip('"')
    if role not in VIEW_ROLE_DESCRIPTIONS:
        allowed = ", ".join(VIEW_ROLE_DESCRIPTIONS.keys())
        raise argparse.ArgumentTypeError(f"unknown view role '{role}'. Allowed: {allowed}")
    if not path:
        raise argparse.ArgumentTypeError("empty image path in --view")
    return role, path


def collect_views(args):
    views = []

    if args.view:
        views.extend(args.view)

    explicit = [
        ("sketch", args.sketch),
        ("canonical_render", args.canonical_render),
        ("front", args.front),
        ("front_left_45", args.front_left_45),
        ("front_right_45", args.front_right_45),
        ("semi_overhead", args.semi_overhead),
    ]
    for role, path in explicit:
        if path:
            views.append((role, path))

    if args.image:
        views.append((args.view_name, args.image))

    # stable de-duplication by (role,path)
    out = []
    seen = set()
    for role, path in views:
        key = (role, os.path.abspath(path))
        if key not in seen:
            out.append((role, path))
            seen.add(key)
    return out


def build_messages(views, voice):
    role_lines = []
    for role, path in views:
        role_lines.append(f"- {role}: {os.path.basename(path)} :: {VIEW_ROLE_DESCRIPTIONS.get(role, '')}")

    user_text = (
        "Input view roles:\n"
        + "\n".join(role_lines)
        + "\n\nDirector's voice transcript:\n---\n"
        + (voice or "")
        + "\n---\n\n"
        "Extract one unified InkVox Scene JSON v0.4 for this same stage scene. "
        "First perform the mandatory visual-only structure audit from the system prompt, ignoring ordinary voice shape/count claims unless the voice explicitly requests an override. "
        "Apply the false-certainty guard: if the footprint is underconstrained by a single perspective, mark it uncertain or composite instead of forcing a full circle from ordinary voice. "
        "Then write scene.metadata.visual_structure_audit and make the final objects follow that audit. "
        "Ground visible structure, footprint, count, and layout primarily in the images. "
        "Use the voice as auxiliary semantic/material/intent evidence unless it explicitly says to modify or override the image. "
        "Use object_groups for repeated spotlights, monitor wedges, side speakers, screen modules, chairs, etc. "
        "Use geometry.representation, source_evidence, relations, constraints and validation fields. "
        "Return only JSON."
    )

    content = [{"type": "text", "text": user_text}]
    for role, path in views:
        content.append({"type": "text", "text": f"VIEW_ROLE={role}\n{VIEW_ROLE_DESCRIPTIONS.get(role, '')}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(path)}})

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def call_vlm(base_url, model, api_key, messages, no_proxy=False, timeout=180):
    import requests
    session = requests.Session()
    proxies = None
    if no_proxy:
        session.trust_env = False
        proxies = {"http": None, "https": None}
    resp = session.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 12000},
        proxies=proxies,
        timeout=timeout,
    )
    if not resp.ok:
        raise SystemExit(
            f"[API error] {resp.status_code} {resp.reason}\n"
            f"Server said: {resp.text[:1500]}"
        )
    return resp.json()["choices"][0]["message"]["content"]


def parse_model_text(text):
    """Extract the JSON object from the model's reply, tolerating prose/code fences."""
    if not text:
        raise ValueError("empty model output")
    text = text.strip()
    # Remove common markdown fences before brace extraction.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("No JSON object found in the model output.")
    return json.loads(text[i:j + 1])



def schema_sanitize_if_available(data):
    """Use the validator's sanitizer when present.

    The extractor is the VLM boundary, so model output can contain almost-right
    v0.4 fields. The validator remains the authoritative repair step. When the
    v0.4 sanitizer is available locally, we reuse it here so scene.raw.json is
    already schema-clean enough for early inspection.
    """
    try:
        from structure_validator import sanitize_scene_json
        return sanitize_scene_json(data)
    except Exception:
        # Conservative fallback for older validator versions.
        if isinstance(data, dict):
            data["schema_version"] = "0.4"
            scene = data.setdefault("scene", {})
            scene["coordinate_system"] = {
                "units": "m", "angle_units": "deg",
                "origin": "downstage_center_floor",
                "x": "audience_right", "y": "upstage_depth", "z": "up",
            }
            val = scene.setdefault("validation", {})
            if val.get("status") not in {"raw", "validated", "has_conflicts", "partial"}:
                val["status"] = "raw"
            val.pop("notes", None)
        return data


def _ensure_list(obj, key):
    val = obj.get(key)
    if isinstance(val, list):
        return val
    if val is None:
        obj[key] = []
    else:
        obj[key] = [val]
    return obj[key]


def _material_source_to_evidence(src):
    if not src:
        return "inferred"
    return src


def voice_has_explicit_override(voice_text):
    """Heuristic marker for voice instructions that intentionally override images.

    This does not decide geometry by itself; it records an evidence flag for the
    validator and prompt-debugging.  Final conflict resolution remains evidence-
    based and schema-compatible.
    """
    if not voice_text:
        return False
    v = str(voice_text).lower()
    override_markers = [
        "改成", "替换成", "不要", "删除", "去掉", "最终只保留", "只保留", "以我说的为准",
        "实际我要", "纠正一下", "按我的描述", "不是图里的", "图里错了", "覆盖图像",
        "change to", "replace with", "remove", "delete", "do not use", "keep only",
        "override", "use my description", "actually i want", "correct it to", "ignore the image",
    ]
    return any(m in v for m in override_markers)



def _voice_mentions_circular_stage(voice_text):
    t = str(voice_text or "").lower()
    return any(k in t for k in ["圆形", "圆的", "圆舞台", "circular", "round stage", "circle stage"])

def _single_view_footprint_guard(scene, voice, views):
    """Record a warning when a single-view extraction overclaims a circular footprint.

    This does not hard-rewrite geometry, because true circular stages exist.
    It gives structure_validator.py and evaluation scripts an explicit signal
    that the circle came from an underconstrained single perspective and should
    be treated cautiously when no semi-overhead evidence is present.
    """
    if not isinstance(scene, dict):
        return
    roles = {r for r, _ in (views or [])}
    has_plan_view = bool(roles & {"semi_overhead"})
    if has_plan_view or len(roles) == 0:
        return
    if not _voice_mentions_circular_stage(voice) or voice_has_explicit_override(voice):
        return

    md = scene.setdefault("metadata", {})
    audit = md.setdefault("visual_structure_audit", {})
    sfv = audit.get("stage_footprint_visual") or {}
    candidate = str(sfv.get("candidate", "")).lower()
    if candidate not in {"circle", "circular_stage"}:
        return

    # Find a main platform with circle footprint.
    platform_ids = []
    for obj in scene.get("objects", []) or []:
        if not isinstance(obj, dict) or obj.get("category") != "platform":
            continue
        geom = obj.get("geometry") or {}
        fp = geom.get("footprint") or {}
        subtype = str(obj.get("subtype", "")).lower()
        shape = str(geom.get("shape", "")).lower()
        if fp.get("type") == "circle" or "circular" in subtype or shape == "circle":
            platform_ids.append(obj.get("id", "platform"))

    if not platform_ids:
        return

    # Treat this as an underconstrained single-view footprint, not as a
    # high-confidence full circle.  We keep the raw geometry so downstream tools
    # remain usable, but lower confidence and add explicit unresolved/conflict
    # records for validator/evaluation.
    sfv["confidence"] = min(float(sfv.get("confidence", 0.6) or 0.6), 0.45)
    sfv["candidate_raw"] = candidate
    sfv["candidate"] = "unknown"
    audit.setdefault("stage_footprint_resolution", {})
    audit["stage_footprint_resolution"]["chosen"] = "unknown"
    audit["stage_footprint_resolution"]["reason"] = "single_view_footprint_underconstrained"

    for obj in scene.get("objects", []) or []:
        if not isinstance(obj, dict) or obj.get("id") not in platform_ids:
            continue
        obj["confidence"] = min(float(obj.get("confidence", 0.6) or 0.6), 0.45)
        geom = obj.get("geometry") if isinstance(obj.get("geometry"), dict) else {}
        fp = geom.get("footprint") if isinstance(geom.get("footprint"), dict) else {}
        if fp:
            fp["confidence"] = min(float(fp.get("confidence", 0.6) or 0.6), 0.45)
            fp.setdefault("source_views", [r for r, _ in (views or [])])
        ev = obj.setdefault("source_evidence", [])
        if isinstance(ev, list):
            ev.append({
                "source": "validator",
                "supports": ["geometry.footprint"],
                "observed_value": "circle_underconstrained_by_single_view",
                "confidence": 0.45,
                "notes": "No semi_overhead/plan view confirms the full closed circular perimeter; ordinary voice mentions circular stage but is not an override."
            })
    sfv.setdefault("visual_cues", [])
    if isinstance(sfv["visual_cues"], list):
        sfv["visual_cues"].append("single-view footprint guard: no semi_overhead view; rear/side boundary may be underconstrained")
    audit["stage_footprint_visual"] = sfv
    audit.setdefault("stage_footprint_resolution", {}).setdefault("reason", "single_view_circle_underconstrained")

    validation = scene.setdefault("validation", {})
    conflicts = validation.setdefault("conflicts", [])
    if not any(isinstance(c, dict) and c.get("field") == "main_stage.geometry.footprint" and "underconstrained" in str(c.get("message", "")).lower() for c in conflicts):
        conflicts.append({
            "type": "ambiguous_type",
            "severity": "warning",
            "field": "main_stage.geometry.footprint",
            "message": "A full circular stage footprint was produced from a single non-plan view while ordinary voice mentions a circular stage. Treat as underconstrained unless a semi-overhead/plan view confirms the full perimeter.",
            "resolution": "keep_raw_output_but_mark_low_confidence_for_validator",
        })
    unresolved = validation.setdefault("unresolved_fields", [])
    if "main_stage.geometry.footprint_requires_plan_view_confirmation" not in unresolved:
        unresolved.append("main_stage.geometry.footprint_requires_plan_view_confirmation")



def postprocess_scene(data, voice, model, views):
    """Light cleanup to make VLM output easier for validator.

    This is intentionally conservative. The validator remains the authority for
    filling positions/sizes and expanding groups.
    """
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")

    data["schema_version"] = "0.4"
    scene = data.setdefault("scene", {})
    scene.setdefault("id", "scene_01")
    scene.setdefault("name", "InkVox extracted scene")
    scene.setdefault("stage", {"width": 12, "depth": 8, "height_clearance": 6})
    stage = scene.setdefault("stage", {})
    stage.setdefault("width", 12)
    stage.setdefault("depth", 8)
    stage.setdefault("height_clearance", 6)

    scene.setdefault("coordinate_system", {
        "units": "m",
        "angle_units": "deg",
        "origin": "downstage_center_floor",
        "x": "audience_right",
        "y": "upstage_depth",
        "z": "up",
    })

    objects = _ensure_list(scene, "objects")
    groups = _ensure_list(scene, "object_groups")
    relations = _ensure_list(scene, "relations")
    constraints = _ensure_list(scene, "constraints")
    source_evidence = _ensure_list(scene, "source_evidence")

    # Basic id/category cleanup and relation migration from placement.
    valid_rel_keys = {"attached_to", "mounted_on", "target", "on_top_of", "behind", "in_front_of"}
    seen_ids = set()
    for idx, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        oid = obj.get("id") or f"object_{idx+1}"
        oid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(oid)).strip("_").lower() or f"object_{idx+1}"
        base = oid
        k = 2
        while oid in seen_ids:
            oid = f"{base}_{k}"
            k += 1
        obj["id"] = oid
        seen_ids.add(oid)
        obj.setdefault("confidence", 0.6)
        obj.setdefault("source", "multi_view" if len(views) > 1 else views[0][0] if views else "inferred")
        obj.setdefault("source_views", [role for role, _ in views] or ["unknown"])
        obj.setdefault("source_evidence", [])

        placement = obj.get("placement")
        if isinstance(placement, dict):
            for key in list(valid_rel_keys):
                target = placement.get(key)
                if not target:
                    continue
                rtype = {"target": "aimed_at"}.get(key, key)
                # avoid adding if already present
                if not any(r.get("type") == rtype and r.get("subject") == oid and r.get("target") == target
                           for r in relations if isinstance(r, dict)):
                    relations.append({
                        "type": rtype,
                        "subject": oid,
                        "target": target,
                        "source": "validator",
                        "confidence": obj.get("confidence", 0.6),
                    })

    for gidx, group in enumerate(groups):
        if not isinstance(group, dict):
            continue
        gid = group.get("id") or f"object_group_{gidx+1}"
        gid = re.sub(r"[^a-zA-Z0-9_]+", "_", str(gid)).strip("_").lower() or f"object_group_{gidx+1}"
        group["id"] = gid
        group.setdefault("confidence", 0.65)
        group.setdefault("source_evidence", [])
        proto = group.setdefault("prototype", {})
        # Do not inject arbitrary fields into prototype: v0.4 schema keeps
        # prototype intentionally narrow. Confidence/evidence belongs on the
        # object_group or inside allowed prototype subfields.

    # Always record source roles so later evaluation can explain where evidence came from.
    visual_roles = {"canonical_render", "front", "front_left_45", "front_right_45", "semi_overhead", "sketch", "other"}
    for role, path in views:
        if not any(ev.get("source") == role and ev.get("notes") == os.path.basename(path)
                   for ev in source_evidence if isinstance(ev, dict)):
            supports = ["visible_structure", "visible_count", "layout", "geometry.footprint"] if role in visual_roles else ["scene_evidence"]
            conf = 0.85 if role in {"canonical_render", "front", "front_left_45", "front_right_45", "semi_overhead"} else 0.45 if role == "sketch" else 0.70
            source_evidence.append({
                "source": role,
                "supports": supports,
                "confidence": conf,
                "notes": os.path.basename(path),
            })

    # Voice evidence is preserved but classified as auxiliary unless it contains
    # explicit edit/override language.
    voice_override = voice_has_explicit_override(voice)
    if voice and not any(ev.get("source") == "voice" and ev.get("source_id") == "voice_transcript"
                         for ev in source_evidence if isinstance(ev, dict)):
        source_evidence.append({
            "source": "voice",
            "source_id": "voice_transcript",
            "supports": ["semantic_labels", "materials", "functions", "user_intent"] + (["explicit_override"] if voice_override else []),
            "observed_value": voice[:500],
            "confidence": 0.85 if voice_override else 0.60,
            "notes": "Voice is auxiliary for visible structure unless it contains an explicit override instruction.",
        })

    # Generic constraints: preserve visual consistency and avoid hallucination.
    if not any(c.get("type") == "visual_consistency" for c in constraints if isinstance(c, dict)):
        constraints.append({
            "type": "visual_consistency",
            "message": "Ground visible object inventory, count, footprint, and spatial layout primarily in image evidence unless voice explicitly overrides it.",
            "priority": "high",
            "source": "validator",
            "confidence": 0.95,
        })
    if not any(c.get("type") == "avoid_new_objects" for c in constraints if isinstance(c, dict)):
        constraints.append({
            "type": "avoid_new_objects",
            "message": "Do not add audience, performers, fog, decorative props, or extra furniture unless explicitly visible or requested.",
            "priority": "high",
            "source": "validator",
            "confidence": 0.95,
        })
    if voice_override and not any(c.get("type") == "user_intent" and "override" in c.get("message", "").lower()
                                  for c in constraints if isinstance(c, dict)):
        constraints.append({
            "type": "user_intent",
            "message": "Voice transcript contains an explicit override/edit cue; validator should consider voice override only for the affected fields.",
            "priority": "high",
            "source": "voice",
            "confidence": 0.85,
        })

    scene.setdefault("validation", {})
    validation = scene["validation"]
    validation.setdefault("status", "raw")
    validation.setdefault("conflicts", [])
    validation.setdefault("unresolved_fields", [])
    validation.setdefault("filled", [])

    md = scene.setdefault("metadata", {})
    md.setdefault("source_voice", voice)
    md.setdefault("model", model)
    md.setdefault("extractor_version", INKVOX_EXTRACTOR_VERSION)
    md.setdefault("input_views", [{"role": role, "path": os.path.basename(path)} for role, path in views])
    if len(views) == 1:
        md.setdefault("input_view_name", views[0][0])
    else:
        md.setdefault("input_view_name", "multi_view")
    md.setdefault("schema_target", "0.4")
    md.setdefault("visual_priority_policy", {
        "visible_structure": "image_primary",
        "voice_role": "auxiliary_for_semantics_materials_functions_and_explicit_overrides",
        "conflict_default": "use_visual_evidence_unless_voice_explicitly_overrides",
        "voice_explicit_override_detected": voice_has_explicit_override(voice),
    })
    md.setdefault("visual_structure_audit", {
        "stage_footprint_visual": {
            "candidate": "unknown",
            "visual_cues": [],
            "confidence": 0.0,
        },
        "stage_footprint_voice": {
            "claim": "",
            "is_explicit_override": voice_has_explicit_override(voice),
        },
        "stage_footprint_resolution": {
            "chosen": "unknown",
            "reason": "model_did_not_provide_audit",
        },
        "visible_count_audit": [],
        "side_structure_audit": {
            "side_walls_visible": None,
            "side_speakers_visible": None,
            "notes": "",
        },
    })
    md.setdefault("notes", "Raw v0.4 scene produced by extract_scene.py; image evidence is primary for visible structure; run structure_validator.py next.")

    _single_view_footprint_guard(scene, voice, views)

    return data


def schema_check(data, schema_path):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:
        print("[note] jsonschema not installed; skipping schema check.")
        return
    if not os.path.isabs(schema_path) and not os.path.exists(schema_path):
        schema_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),schema_path)
    if not os.path.exists(schema_path):
        print(f"[note] {schema_path} not found; skipping schema check.")
        return
    schema = json.load(open(schema_path, encoding="utf-8"))
    schema_ver = schema.get("properties", {}).get("schema_version", {}).get("const")
    data_ver = data.get("schema_version")
    if schema_ver and data_ver and str(schema_ver) != str(data_ver):
        print(f"[warn] schema version mismatch: output is {data_ver}, schema expects {schema_ver}.")
    errs = sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path))
    if not errs:
        print("[ok] output conforms to the schema (raw state).")
        return
    print(f"[warn] output has {len(errs)} schema issue(s):")
    for e in errs[:20]:
        where = " -> ".join(str(p) for p in e.path) or "(root)"
        print(f"   at {where} :: {e.message}")


# ==================================================================== main ==
def list_models(base_url, api_key, no_proxy=False):
    import requests
    session = requests.Session()
    proxies = None
    if no_proxy:
        session.trust_env = False
        proxies = {"http": None, "https": None}
    resp = session.get(
        f"{base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        proxies=proxies, timeout=60,
    )
    if not resp.ok:
        raise SystemExit(f"[API error] {resp.status_code} {resp.reason}\nServer said: {resp.text[:1000]}")
    ids = [m.get("id", "") for m in resp.json().get("data", [])]
    vl = sorted(i for i in ids if "vl" in i.lower() or "vision" in i.lower())
    print(f"{len(ids)} model(s) available to this key.")
    print("\nVision-language models you can use:")
    for i in (vl or ["  (none found - filter the full list below for a vision model)"]):
        print("  " + i)
    if not vl:
        print("\nAll model ids:")
        for i in sorted(ids):
            print("  " + i)


def main():
    ap = argparse.ArgumentParser(description="InkVox images + voice -> raw Scene JSON v0.4 via a VLM.")

    # Backward-compatible single image.
    ap.add_argument("--image", help="single sketch/rendering image path")
    ap.add_argument("--view-name", default="sketch",
                    choices=list(VIEW_ROLE_DESCRIPTIONS.keys()),
                    help="role for --image; for multi-view prefer --view role=path")

    # Multi-view input.
    ap.add_argument("--view", action="append", type=parse_view_arg, default=[],
                    help="add a view as role=path. Roles: sketch, canonical_render, front, front_left_45, front_right_45, semi_overhead, other")
    ap.add_argument("--sketch", default=None, help="path to original sketch")
    ap.add_argument("--canonical-render", dest="canonical_render", default=None, help="path to first/main generated render")
    ap.add_argument("--front", default=None, help="path to front/audience render")
    ap.add_argument("--front-left-45", dest="front_left_45", default=None, help="path to front-left 45-degree render")
    ap.add_argument("--front-right-45", dest="front_right_45", default=None, help="path to front-right 45-degree render")
    ap.add_argument("--semi-overhead", dest="semi_overhead", default=None, help="path to semi-overhead/elevated render")

    ap.add_argument("--voice", default="", help="voice transcript text")
    ap.add_argument("--voice-file", default=None, help="read transcript from a text file")
    ap.add_argument("--out", default="scene.raw.json", help="output raw v0.4 scene path")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct",
                    help="VLM model id; avoid Thinking variants because they can pollute JSON")
    ap.add_argument("--base-url", default="https://api.siliconflow.cn/v1",
                    help="OpenAI-compatible base URL")
    ap.add_argument("--api-key-env", default="SILICONFLOW_API_KEY",
                    help="name of env var holding the API key")
    ap.add_argument("--schema", default="scene.schema.v0.4.json", help="schema for the post-check")
    ap.add_argument("--from-text", default=None,
                    help="DEBUG: read model output from this file instead of calling the API")
    ap.add_argument("--dump-prompt", default=None,
                    help="DEBUG: write the multimodal message text/roles to this file and exit before API")
    ap.add_argument("--no-proxy", action="store_true",
                    help="bypass all proxies (env vars AND Windows system proxy) for the API call")
    ap.add_argument("--timeout", type=int, default=240,
                    help="read timeout in seconds")
    ap.add_argument("--list-models", action="store_true",
                    help="list models this key can use, then exit")
    args = ap.parse_args()

    if args.list_models:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Error: no API key in env var {args.api_key_env}.")
            sys.exit(2)
        list_models(args.base_url, api_key, no_proxy=args.no_proxy)
        return

    voice = args.voice
    if args.voice_file:
        voice = open(args.voice_file, encoding="utf-8").read().strip()

    views = collect_views(args)

    if args.dump_prompt:
        # Useful for checking front-end Print prompt wording without exposing base64.
        payload = {
            "extractor_version": INKVOX_EXTRACTOR_VERSION,
            "views": [{"role": r, "path": p, "description": VIEW_ROLE_DESCRIPTIONS.get(r)} for r, p in views],
            "voice": voice,
            "system_prompt": SYSTEM_PROMPT,
        }
        with open(args.dump_prompt, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote prompt debug -> {args.dump_prompt}")
        return

    if args.from_text:
        raw_text = open(args.from_text, encoding="utf-8").read()
    else:
        if not views:
            print("Error: provide --image or one/more --view role=path inputs (unless using --from-text).")
            sys.exit(2)
        for role, path in views:
            if not os.path.exists(path):
                print(f"Error: image for role '{role}' not found: {path}")
                sys.exit(2)
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            print(f"Error: no API key. Set it first, e.g.:\n"
                  f'  setx {args.api_key_env} "sk-your-key"   (PowerShell, then reopen terminal)')
            sys.exit(2)
        print(f"[extract] version={INKVOX_EXTRACTOR_VERSION}")
        print(f"[extract] model={args.model}")
        print("[extract] views=" + ", ".join(f"{r}:{os.path.basename(p)}" for r, p in views))
        messages = build_messages(views, voice)
        raw_text = call_vlm(args.base_url, args.model, api_key, messages,
                            no_proxy=args.no_proxy, timeout=args.timeout)

    try:
        data = parse_model_text(raw_text)
        data = postprocess_scene(data, voice, args.model, views)
        # Model output cannot create user adoption, even if it claims to.
        from decision_state import initialize
        for obj in data.get('scene',{}).get('objects',[]):
            for state in obj.get('parameter_states',{}).values():
                state['adoption_status']='provisional';state['adopted_value']=None
        data = schema_sanitize_if_available(data)
        initialize(data['scene'])
        # Re-apply schema-compatible uncertainty guards after sanitizer, because
        # sanitizer may drop non-schema conflict fields from raw model output.
        if isinstance(data, dict):
            _single_view_footprint_guard(data.get("scene", {}), voice, views)
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Could not parse/postprocess JSON from the model output: {e}")
        print("---- raw model output ----")
        print(raw_text[:3000])
        sys.exit(1)

    schema_check(data, args.schema)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    scene = data.get("scene", {})
    n_obj = len(scene.get("objects", []))
    n_grp = len(scene.get("object_groups", []))
    n_rel = len(scene.get("relations", []))
    print(f"Wrote -> {args.out}  ({n_obj} object(s), {n_grp} group(s), {n_rel} relation(s))")
    print("Next:  python structure_validator.py " + args.out + " -o scene.filled.json")


if __name__ == "__main__":
    main()
