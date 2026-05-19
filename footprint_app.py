import os
import re
import tempfile
import math

import numpy as np
from flask import Flask, render_template, request, jsonify
from stl import mesh as stl_mesh
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXT = {".stl", ".step", ".stp"}
GAP_DEFAULT = 2.0
LARGE_MESH_THRESHOLD = 150_000

# Axis index mapping: which two indices define the footprint plane
# and which index is the build height
AXIS_MAP = {
    "z": {"fp": (0, 1), "h": 2, "label": "XY",  "axes": ("X", "Y")},
    "y": {"fp": (0, 2), "h": 1, "label": "XZ",  "axes": ("X", "Z")},
    "x": {"fp": (1, 2), "h": 0, "label": "YZ",  "axes": ("Y", "Z")},
}


# ---------------------------------------------------------------------------
# Core projection helpers
# ---------------------------------------------------------------------------

def _project_triangles(triangles_xyz: np.ndarray, ax: dict):
    """
    True orthographic footprint: union of all XY-projected triangles.
    ax: one of the AXIS_MAP values — defines which two coordinates form the
        footprint plane.
    Returns a Shapely geometry or None.
    """
    i0, i1 = ax["fp"]
    polys = []
    for tri in triangles_xyz:
        pts = [(float(tri[v, i0]), float(tri[v, i1])) for v in range(3)]
        try:
            p = Polygon(pts)
            if p.is_valid and p.area > 1e-10:
                polys.append(p)
        except Exception:
            pass
    return unary_union(polys) if polys else None


def _convex_hull_from_verts(verts_2d: np.ndarray):
    """Convex hull fallback for large meshes / STEP files."""
    unique = np.unique(verts_2d, axis=0)
    if len(unique) < 3:
        return None
    try:
        return MultiPoint(unique).convex_hull
    except Exception:
        return None


def _shapely_to_polygon_data(geom, min0: float, min1: float) -> dict | None:
    """
    Convert a Shapely polygon to the JSON structure the frontend uses.
    Coordinates are normalised to the footprint bounding-box origin.
    """
    def norm(coords):
        return [[round(p[0] - min0, 4), round(p[1] - min1, 4)] for p in coords]

    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "Polygon":
        return {
            "exterior": norm(geom.exterior.coords),
            "holes":    [norm(h.coords) for h in geom.interiors],
            "area":     float(geom.area),
        }
    if geom.geom_type == "MultiPolygon":
        largest = max(geom.geoms, key=lambda g: g.area)
        return {
            "exterior": norm(largest.exterior.coords),
            "holes":    [norm(h.coords) for h in largest.interiors],
            "area":     float(geom.area),
        }
    return None


# ---------------------------------------------------------------------------
# STL analyser
# ---------------------------------------------------------------------------

def analyze_stl(filepath: str, build_axis: str = "z") -> dict:
    ax = AXIS_MAP[build_axis]
    i0, i1, ih = ax["fp"][0], ax["fp"][1], ax["h"]

    your_mesh = stl_mesh.Mesh.from_file(filepath)
    tris  = your_mesh.vectors          # (N, 3, 3)
    verts = tris.reshape(-1, 3)

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)

    bbox_fp0  = float(maxs[i0] - mins[i0])   # footprint width
    bbox_fp1  = float(maxs[i1] - mins[i1])   # footprint depth
    height    = float(maxs[ih] - mins[ih])   # build height
    min0, min1 = float(mins[i0]), float(mins[i1])

    n_tris = len(tris)

    if n_tris <= LARGE_MESH_THRESHOLD:
        footprint = _project_triangles(tris, ax)
        note = (
            f"True orthographic projection onto the {ax['label']} plane — "
            f"union of all {n_tris:,} projected triangles. "
            f"Build axis: {build_axis.upper()}."
        )
    else:
        verts_2d = verts[:, [i0, i1]]
        footprint = _convex_hull_from_verts(verts_2d)
        note = (
            f"Mesh has {n_tris:,} triangles — convex-hull approximation used for speed. "
            f"Build axis: {build_axis.upper()}."
        )

    if footprint is not None and not footprint.is_empty:
        fp_area = float(footprint.area)
        tol = max(0.01, math.sqrt(bbox_fp0**2 + bbox_fp1**2) * 0.001)
        simplified = footprint.simplify(tol, preserve_topology=True)
        poly_data = _shapely_to_polygon_data(simplified, min0, min1)
    else:
        fp_area   = bbox_fp0 * bbox_fp1
        poly_data = None

    return {
        "bbox_x":            round(bbox_fp0, 3),
        "bbox_y":            round(bbox_fp1, 3),
        "height_z":          round(height, 3),
        "bbox_area":         round(bbox_fp0 * bbox_fp1, 2),
        "footprint_area":    round(fp_area, 2),
        "footprint_polygon": poly_data,
        "source_note":       note,
        "vertex_count":      int(len(verts)),
        "build_axis":        build_axis.upper(),
        "fp_axes":           ax["axes"],
    }


# ---------------------------------------------------------------------------
# STEP analyser
# ---------------------------------------------------------------------------

def analyze_step(filepath: str, build_axis: str = "z") -> dict:
    ax = AXIS_MAP[build_axis]
    i0, i1, ih = ax["fp"][0], ax["fp"][1], ax["h"]

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
    mins  = verts.min(axis=0)
    maxs  = verts.max(axis=0)

    bbox_fp0  = float(maxs[i0] - mins[i0])
    bbox_fp1  = float(maxs[i1] - mins[i1])
    height    = float(maxs[ih] - mins[ih])
    min0, min1 = float(mins[i0]), float(mins[i1])

    verts_2d  = verts[:, [i0, i1]]
    footprint = _convex_hull_from_verts(verts_2d)

    if footprint is not None and not footprint.is_empty:
        fp_area   = float(footprint.area)
        poly_data = _shapely_to_polygon_data(footprint, min0, min1)
    else:
        fp_area   = bbox_fp0 * bbox_fp1
        poly_data = None

    return {
        "bbox_x":            round(bbox_fp0, 3),
        "bbox_y":            round(bbox_fp1, 3),
        "height_z":          round(height, 3),
        "bbox_area":         round(bbox_fp0 * bbox_fp1, 2),
        "footprint_area":    round(fp_area, 2),
        "footprint_polygon": poly_data,
        "source_note":       (
            f"STEP file: convex hull of extracted vertices onto {ax['label']} plane. "
            f"Build axis: {build_axis.upper()}. "
            "Upload an STL for the exact triangle-level projection."
        ),
        "vertex_count":      int(len(verts)),
        "build_axis":        build_axis.upper(),
        "fp_axes":           ax["axes"],
    }


# ---------------------------------------------------------------------------
# Packing calculator
# ---------------------------------------------------------------------------

def packing_estimate(bbox_x: float, bbox_y: float,
                     bed_x: float, bed_y: float, gap: float) -> dict:
    step_x = bbox_x + gap
    step_y = bbox_y + gap
    nx = max(0, math.floor(bed_x / step_x)) if step_x > 0 else 0
    ny = max(0, math.floor(bed_y / step_y)) if step_y > 0 else 0
    total = nx * ny

    used_area = total * bbox_x * bbox_y
    bed_area  = bed_x * bed_y
    fill_pct  = round(used_area / bed_area * 100, 1) if bed_area > 0 else 0

    placements = [
        {"x": col * step_x, "y": row * step_y}
        for row in range(ny)
        for col in range(nx)
    ]
    return {"nx": nx, "ny": ny, "total": total, "fill_pct": fill_pct,
            "placements": placements}


# ---------------------------------------------------------------------------
# Bed fill assessment
# ---------------------------------------------------------------------------

def bed_assessment(footprint_area: float, bed_x: float, bed_y: float) -> dict:
    bed_area   = bed_x * bed_y
    single_pct = round(footprint_area / bed_area * 100, 1) if bed_area > 0 else 0

    if single_pct >= 80:
        level, message, color = "full",     "Bed is essentially full with this single part.", "red"
    elif single_pct >= 50:
        level, message, color = "crowded",  "More than half the bed is occupied. Limited room for additional parts.", "amber"
    elif single_pct >= 20:
        level, message, color = "moderate", "Reasonable space available. More parts can be added.", "blue"
    else:
        level, message, color = "spacious", "Plenty of bed space remaining. Good candidate for batch printing.", "green"

    return {"level": level, "message": message, "color": color,
            "single_part_pct": single_pct}


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

    bed_x      = float(request.form.get("bed_x", 250))
    bed_y      = float(request.form.get("bed_y", 250))
    gap        = float(request.form.get("gap", GAP_DEFAULT))
    build_axis = request.form.get("build_axis", "z").lower()
    if build_axis not in AXIS_MAP:
        build_axis = "z"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        geo = (analyze_stl(tmp_path, build_axis)
               if ext == ".stl"
               else analyze_step(tmp_path, build_axis))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 422
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    pack       = packing_estimate(geo["bbox_x"], geo["bbox_y"], bed_x, bed_y, gap)
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
