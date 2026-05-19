import os
import re
import tempfile
import math

import numpy as np
import trimesh
from trimesh.path.polygons import projected as trimesh_projected
from flask import Flask, render_template, request, jsonify
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXT = {".stl", ".step", ".stp"}
GAP_DEFAULT = 2.0


# ---------------------------------------------------------------------------
# Polygon -> JSON helper
# ---------------------------------------------------------------------------

def _shapely_to_polygon_data(geom, x_min: float, y_min: float):
    """
    Convert a Shapely polygon to the JSON structure the frontend expects.
    Coordinates are returned relative to (x_min, y_min) so the frontend can
    position the part anywhere on the bed.
    """
    def norm(coords):
        return [[round(p[0] - x_min, 4), round(p[1] - y_min, 4)] for p in coords]

    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return {
            "exterior": norm(geom.exterior.coords),
            "holes":    [norm(h.coords) for h in geom.interiors],
            "area":     float(geom.area),
        }
    if geom.geom_type == "MultiPolygon":
        # Pick the largest polygon for the outline (the user's main part)
        largest = max(geom.geoms, key=lambda g: g.area)
        return {
            "exterior": norm(largest.exterior.coords),
            "holes":    [norm(h.coords) for h in largest.interiors],
            "area":     float(geom.area),   # total of all parts
        }
    return None


# ---------------------------------------------------------------------------
# STL analyser — uses trimesh for robust loading + projected outline
# ---------------------------------------------------------------------------

def analyze_stl(filepath: str) -> dict:
    """
    Load any STL file (binary, ASCII, multi-solid, scene) via trimesh,
    then compute the true XY orthographic projection looking down -Z.
    """
    loaded = trimesh.load(filepath, force="mesh")

    if hasattr(loaded, "geometry") and not isinstance(loaded, trimesh.Trimesh):
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No mesh geometry found in STL.")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError("STL file contains no triangle data.")

    verts = np.asarray(mesh.vertices)
    n_tris = len(mesh.triangles)
    n_verts = len(verts)

    # Bounding box from ALL vertices — guaranteed correct
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    bbox_x = float(maxs[0] - mins[0])
    bbox_y = float(maxs[1] - mins[1])
    height = float(maxs[2] - mins[2])
    x_min, y_min = float(mins[0]), float(mins[1])

    # True XY footprint: trimesh handles silhouette projection robustly,
    # including concave shapes, holes and complex/disjoint meshes.
    try:
        footprint = trimesh_projected(mesh, normal=[0.0, 0.0, 1.0])
    except Exception:
        footprint = None

    if footprint is None or footprint.is_empty:
        # Last-resort fallback so we still return something useful
        footprint = MultiPoint(verts[:, :2]).convex_hull
        source_note = (
            f"Loaded {n_verts:,} vertices · {n_tris:,} triangles. "
            "Silhouette projection failed; showing convex hull of vertex cloud."
        )
    else:
        source_note = (
            f"True orthographic projection onto the XY plane (looking down -Z). "
            f"Loaded {n_verts:,} vertices, {n_tris:,} triangles."
        )

    fp_area = float(footprint.area)
    # Simplify outline for SVG rendering (tolerance: 0.1% of bbox diagonal)
    tol = max(0.01, math.sqrt(bbox_x**2 + bbox_y**2) * 0.001)
    simplified = footprint.simplify(tol, preserve_topology=True)
    poly_data = _shapely_to_polygon_data(simplified, x_min, y_min)

    return {
        "bbox_x":            round(bbox_x, 3),
        "bbox_y":            round(bbox_y, 3),
        "height_z":          round(height, 3),
        "bbox_area":         round(bbox_x * bbox_y, 2),
        "footprint_area":    round(fp_area, 2),
        "footprint_polygon": poly_data,
        "source_note":       source_note,
        "vertex_count":      int(n_verts),
        "triangle_count":    int(n_tris),
    }


# ---------------------------------------------------------------------------
# STEP analyser — convex hull of extracted ASCII CARTESIAN_POINTs
# ---------------------------------------------------------------------------

def analyze_step(filepath: str) -> dict:
    try:
        with open(filepath, "r", errors="ignore") as fh:
            content = fh.read()
    except Exception as exc:
        raise ValueError(f"Cannot read STEP file: {exc}") from exc

    if "BINARY" in content[:200].upper():
        raise ValueError("Binary STEP files are not supported. Please export as ASCII STEP.")

    pattern = re.compile(
        r"CARTESIAN_POINT\s*\([^,]*,\s*\(\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
        re.IGNORECASE,
    )
    coords = [(float(m.group(1)), float(m.group(2)), float(m.group(3)))
              for m in pattern.finditer(content)]
    if not coords:
        raise ValueError(
            "No geometry data found. The file may be empty, corrupt, "
            "or not a valid STEP AP203/AP214/AP242 file."
        )

    verts = np.array(coords, dtype=float)
    mins, maxs = verts.min(axis=0), verts.max(axis=0)
    bbox_x = float(maxs[0] - mins[0])
    bbox_y = float(maxs[1] - mins[1])
    height = float(maxs[2] - mins[2])
    x_min, y_min = float(mins[0]), float(mins[1])

    unique = np.unique(verts[:, :2], axis=0)
    footprint = MultiPoint(unique).convex_hull if len(unique) >= 3 else None

    if footprint is not None and not footprint.is_empty:
        fp_area = float(footprint.area)
        poly_data = _shapely_to_polygon_data(footprint, x_min, y_min)
    else:
        fp_area = bbox_x * bbox_y
        poly_data = None

    return {
        "bbox_x":            round(bbox_x, 3),
        "bbox_y":            round(bbox_y, 3),
        "height_z":          round(height, 3),
        "bbox_area":         round(bbox_x * bbox_y, 2),
        "footprint_area":    round(fp_area, 2),
        "footprint_polygon": poly_data,
        "source_note":       (
            f"STEP file: convex hull of {len(verts):,} extracted vertices on the XY plane. "
            "Upload an STL for the exact triangle-level silhouette."
        ),
        "vertex_count":      int(len(verts)),
        "triangle_count":    0,
    }


# ---------------------------------------------------------------------------
# Packing + assessment
# ---------------------------------------------------------------------------

def packing_estimate(bbox_x, bbox_y, bed_x, bed_y, gap):
    step_x = bbox_x + gap
    step_y = bbox_y + gap
    nx = max(0, math.floor(bed_x / step_x)) if step_x > 0 else 0
    ny = max(0, math.floor(bed_y / step_y)) if step_y > 0 else 0
    total = nx * ny

    used_area = total * bbox_x * bbox_y
    bed_area = bed_x * bed_y
    fill_pct = round(used_area / bed_area * 100, 1) if bed_area > 0 else 0
    placements = [{"x": c * step_x, "y": r * step_y} for r in range(ny) for c in range(nx)]
    return {"nx": nx, "ny": ny, "total": total, "fill_pct": fill_pct, "placements": placements}


def bed_assessment(fp_area, bed_x, bed_y):
    bed_area = bed_x * bed_y
    pct = round(fp_area / bed_area * 100, 1) if bed_area > 0 else 0
    if pct >= 80:
        level, msg, color = "full", "Bed is essentially full with this single part.", "red"
    elif pct >= 50:
        level, msg, color = "crowded", "More than half the bed is occupied. Limited room for additional parts.", "amber"
    elif pct >= 20:
        level, msg, color = "moderate", "Reasonable space available. More parts can be added.", "blue"
    else:
        level, msg, color = "spacious", "Plenty of bed space remaining. Good candidate for batch printing.", "green"
    return {"level": level, "message": msg, "color": color, "single_part_pct": pct}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("footprint.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format '{ext}'. Upload .stl, .step, or .stp."}), 400

    bed_x = float(request.form.get("bed_x", 250))
    bed_y = float(request.form.get("bed_y", 250))
    gap   = float(request.form.get("gap",   GAP_DEFAULT))

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        geo = analyze_stl(tmp_path) if ext == ".stl" else analyze_step(tmp_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    pack = packing_estimate(geo["bbox_x"], geo["bbox_y"], bed_x, bed_y, gap)
    assessment = bed_assessment(geo["footprint_area"], bed_x, bed_y)

    return jsonify({
        "filename":   f.filename,
        "geometry":   geo,
        "packing":    pack,
        "assessment": assessment,
        "bed":        {"x": bed_x, "y": bed_y},
        "gap":        gap,
    })


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 54)
    print("  LPBF Bed Footprint Calculator")
    print("  Open your browser at:  http://localhost:5001")
    print("=" * 54)
    print()
    app.run(debug=True, host="0.0.0.0", port=5001)
