import os
import re
import tempfile
import math

import numpy as np
from flask import Flask, render_template, request, jsonify
from stl import mesh as stl_mesh
from shapely.geometry import MultiPoint

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

ALLOWED_EXT = {".stl", ".step", ".stp"}
GAP_DEFAULT = 2.0  # mm spacing between parts


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vertices_to_result(verts: np.ndarray, source_note: str) -> dict:
    """Given an (N,3) array of vertices compute footprint metrics."""
    if len(verts) == 0:
        raise ValueError("No geometry found in file.")

    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)

    bbox_x = float(x_max - x_min)
    bbox_y = float(y_max - y_min)
    height_z = float(z_max - z_min)

    # Convex hull footprint on XY plane
    pts_2d = verts[:, :2]
    unique_pts = np.unique(pts_2d, axis=0)
    hull_area = None
    hull_polygon = None
    if len(unique_pts) >= 3:
        try:
            mp = MultiPoint(unique_pts)
            hull = mp.convex_hull
            hull_area = float(hull.area)
            coords = list(hull.exterior.coords)
            # Normalise to start from (x_min, y_min)
            hull_polygon = [[float(p[0] - x_min), float(p[1] - y_min)] for p in coords]
        except Exception:
            pass

    if hull_area is None:
        hull_area = bbox_x * bbox_y

    return {
        "bbox_x": round(bbox_x, 3),
        "bbox_y": round(bbox_y, 3),
        "height_z": round(height_z, 3),
        "bbox_area": round(bbox_x * bbox_y, 2),
        "hull_area": round(hull_area, 2),
        "hull_polygon": hull_polygon,
        "source_note": source_note,
        "vertex_count": len(verts),
    }


def analyze_stl(filepath: str) -> dict:
    your_mesh = stl_mesh.Mesh.from_file(filepath)
    # vectors shape: (N_triangles, 3_vertices, 3_xyz)
    verts = your_mesh.vectors.reshape(-1, 3)
    return _vertices_to_result(verts, "Full mesh — exact convex hull footprint")


def analyze_step(filepath: str) -> dict:
    """
    Extract CARTESIAN_POINT data from a STEP ASCII file.
    Returns bounding box; convex hull is computed when enough points exist.
    Note: binary STEP is not supported via this method.
    """
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
    except Exception as exc:
        raise ValueError(f"Cannot read STEP file: {exc}") from exc

    # Check for binary STEP (rare but exists)
    if content.startswith("ISO-10303") and "BINARY" in content[:200].upper():
        raise ValueError(
            "Binary STEP files are not supported. Please export as ASCII STEP."
        )

    # Extract CARTESIAN_POINT( '', (x, y, z) ) entries — AP203 / AP214 / AP242
    pattern = re.compile(
        r"CARTESIAN_POINT\s*\([^,]*,\s*\(\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)",
        re.IGNORECASE,
    )
    coords = [
        (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        for m in pattern.finditer(content)
    ]

    if not coords:
        raise ValueError(
            "No geometry data found. The file may be empty, corrupt, "
            "or not a valid STEP AP203/AP214/AP242 file."
        )

    verts = np.array(coords, dtype=float)
    note = (
        "STEP bounding box — convex hull footprint from extracted vertex data. "
        "For highest accuracy use the STL export of this part."
    )
    return _vertices_to_result(verts, note)


# ---------------------------------------------------------------------------
# Packing calculator
# ---------------------------------------------------------------------------

def packing_estimate(bbox_x: float, bbox_y: float, bed_x: float, bed_y: float, gap: float) -> dict:
    """Simple rectangular grid packing with a configurable gap between parts."""
    step_x = bbox_x + gap
    step_y = bbox_y + gap

    nx = max(0, math.floor(bed_x / step_x)) if step_x > 0 else 0
    ny = max(0, math.floor(bed_y / step_y)) if step_y > 0 else 0
    total = nx * ny

    used_area = nx * ny * bbox_x * bbox_y
    bed_area = bed_x * bed_y
    fill_pct = round(used_area / bed_area * 100, 1) if bed_area > 0 else 0

    # Placement grid for SVG rendering (offset relative to bed origin, mm)
    placements = []
    for row in range(ny):
        for col in range(nx):
            placements.append({
                "x": col * step_x,
                "y": row * step_y,
            })

    return {
        "nx": nx,
        "ny": ny,
        "total": total,
        "fill_pct": fill_pct,
        "placements": placements,
    }


# ---------------------------------------------------------------------------
# Bed fill assessment
# ---------------------------------------------------------------------------

def bed_assessment(hull_area: float, bed_x: float, bed_y: float) -> dict:
    bed_area = bed_x * bed_y
    single_pct = round(hull_area / bed_area * 100, 1) if bed_area > 0 else 0

    if single_pct >= 80:
        level = "full"
        message = "Bed is essentially full with this single part."
        color = "red"
    elif single_pct >= 50:
        level = "crowded"
        message = "More than half the bed is occupied. Limited room for additional parts."
        color = "amber"
    elif single_pct >= 20:
        level = "moderate"
        message = "Reasonable space available. More parts can be added."
        color = "blue"
    else:
        level = "spacious"
        message = "Plenty of bed space remaining. Good candidate for batch printing."
        color = "green"

    return {
        "level": level,
        "message": message,
        "color": color,
        "single_part_pct": single_pct,
    }


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
    gap   = float(request.form.get("gap", GAP_DEFAULT))

    # Save to temp file
    suffix = ext
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    try:
        if ext == ".stl":
            geo = analyze_stl(tmp_path)
        else:
            geo = analyze_step(tmp_path)
    except Exception as exc:
        os.unlink(tmp_path)
        return jsonify({"error": str(exc)}), 422
    finally:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass

    pack = packing_estimate(geo["bbox_x"], geo["bbox_y"], bed_x, bed_y, gap)
    assessment = bed_assessment(geo["hull_area"], bed_x, bed_y)

    return jsonify({
        "filename": f.filename,
        "geometry": geo,
        "packing":  pack,
        "assessment": assessment,
        "bed": {"x": bed_x, "y": bed_y},
        "gap": gap,
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
