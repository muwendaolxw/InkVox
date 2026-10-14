# InkVox

InkVox is a research prototype for early-stage stage design. It combines sketch and voice, generates reference views, exposes the resulting interpretation as Scene JSON, and produces editable Blender scenes. Its parameter interfaces distinguish concrete values used for preview from decisions explicitly adopted by the designer.

The project accompanies **InkVox: Preserving Unresolved Design Choices in Multimodal Stage Design**.

## System overview

1. **Express a proposal:** draw or upload a sketch and provide voice or typed input. Users can combine these inputs according to what they need to express.
2. **Generate reference views:** create a reference rendering and four derived views: front, front-left 45 degrees, front-right 45 degrees, and semi-overhead. These generated images are references, not independent geometric observations.
3. **Inspect the interpretation:** use **Print** to obtain Scene JSON. Inspect or download the JSON, or select **Manage decisions** to open the parameter interface.
4. **Revise parameters:** inspect design intent, current values, and recorded sources; add object notes; edit parameters; and save the parameter records.
5. **Validate and generate:** normalize and complete selected fields, check implemented constraints and schema consistency, and construct an editable Blender prototype.

Concrete preview values do not automatically represent user adoption. Adoption also does not establish production feasibility: generated scenes require professional verification before production use.

## Parameter interfaces

The prototype includes two interfaces used in the revision and handoff comparison:

| Interface | Function |
| --- | --- |
| **B: parameters and notes** | Parameter editing, design intent, recorded sources, object notes, saving, preview updates, and export. Designers record decisions and open questions in notes; this interface does not provide explicit adoption operations. |
| **C: explicit decision states** | The common parameter and note functions, plus operations for trying, adopting, retaining provisional values, and reopening decisions. Provisional and adopted states accompany exported parameter records. |

In C, **Try** applies a provisional value; **Adopt** records an explicit decision; **Keep provisional** retains an open choice; and reopening permits an adopted decision to be reconsidered. **Update 3D preview** is separate from adoption. The panel's save control saves parameter records; the main toolbar save control retains its project-saving function.

## Source files

The source files provide the following system components:

| File | Purpose |
| --- | --- |
| `inkvox-stage.html` | Sketch and voice interface, generated-view inspection, and Scene JSON preview. |
| `InkVox_Start.bat` | Windows entry point; calls `launch_inkvox.py`. |
| `extract_scene.py` | Command-line scene extraction through an OpenAI-compatible vision-language endpoint. |
| `structure_validator.py` | Field normalization, selected completion, relationship checks, and decision-state reconciliation. |
| `validate_scene.py` | Schema validation and decision-state consistency checks. |
| `scene.schema.v0.4.json` | Scene JSON v0.4 schema. |
| `generate_stage.py` | Blender scene construction and rendering. |

## Typical workflow

Create a project, configure the model services, and express a proposal through sketch and voice. Generate and inspect the reference views, then click **Print**. This action parses the scene rather than printing a document.

Inspect Scene JSON or choose **Manage decisions**. In the parameter interface, review values and sources, revise selected properties, and add object notes. In C, explicitly distinguish provisional values from adopted decisions. Save the records before exporting. Use **Update 3D preview** to examine the current values in a generated prototype.

Exported scene records and handoff materials support subsequent inspection. The Blender pipeline embeds parameter records, sources, and notes in generated artifacts. Later manual geometry edits in Blender do not automatically synchronize with the browser's parameter records.

## Scope

- Model-generated views and scene interpretations can contain inconsistent or incorrect details.
- Confidence fields are not calibrated uncertainty estimates.
- Validation covers implemented structural and geometric rules; it does not certify structural safety, rigging, equipment performance, construction feasibility, or other production requirements.
- Explicit adoption records a user choice, not professional verification.
- Embedded records do not automatically track later manual Blender geometry edits.
- External API services, model versions, and generation variability can affect reproduction.
