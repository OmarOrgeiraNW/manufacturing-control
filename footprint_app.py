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
# Above this triangle count use convex-hull fallback for speed
LARGE_MESH_THRESHOLD = 150_000


# ---------------------------------------------------------------------------
# Core projection: true orthographic footprint
# ---------------------------------------------------------------------------

def _project_triangles_to_footprint(triangles_xyz: np.ndarray):
    """
    True orthographic footprint: union of all XY projections of every mesh triangle.

    triangles_xyz: shape (N, 3, 3)  — N triangles, 3 vertices, XYZ.

    Vertical faces (zero projected area) are silently skipped.
    Returns a Shapely geometry (Polygon or MultiPolygon), or None if nothing projected.
    """
    polys = []
    for tri in triangles_xyz:
        xy = [(float(tri[0, 0]), float(tri[0, 1])),
              (float(tri[1, 0]), float(tri[1, 1])),
              (float(tri[2, 0]), float(tri[2, 1]))]
        try:
            p = Polygon(xy)
            # Skip degenerate / nearly-vertical faces
            if p.is_valid and p.area > 1e-10:
                polys.append(p)
        except Exception:
            pass

    if not polys:
        return None
    return unary_union(polys)


def _convex_hull_footprint(verts_xy: np.ndarray):
    """Fallback: convex hull of projected vertices (for large meshes / STEP)."""
    unique = np.unique(verts_xy, axis=0)
    if len(unique) < 3:
        return None
    return MultiPoint(unique).convex_hull


def _shapely_to_polygon_data(geom, x_min: float, y_min: float) -> dict:
    """
    Convert a Shapely geometry to the JSON structure the frontend expects.
    Returns {"exterior": [[x,y],...], "holes": [[[x,y],...], ...], "area": float}.
    Coordinates are normalised relative to the part's bounding-box origin.
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
        # Return the largest component — adequate for LPBF parts
        largest = max(geom.geoms, key=lambda g: g.area)
        return {
            "exterior": norm(largest.exterior.coords),
            "holes":    [norm(h.coords) for h in largest.interiors],
            "area":     float(geom.area),   # total, including smaller pieces
        }
    return None


# ---------------------------------------------------------------------------
# STL analyser
# ---------------------------------------------------------------------------

def analyze_stl(filepath: str) -> dict:
    your_mesh = stl_mesh.Mesh.from_file(filepath)
    # shape: (N_triangles, 3_vertices, 3_xyz)
    tris = your_mesh.vectors
    verts = tris.reshape(-1, 3)

    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)
    bbox_x   = float(x_max - x_min)
    bbox_y   = float(y_max - y_min)
    height_z = float(z_max - z_min)

    n_tris = len(tris)

    if n_tris <= LARGE_MESH_THRESHOLD:
        footprint = _project_triangles_to_footprint(tris)
        note = (
            f"True orthographic projection — union of all {n_tris:,} "
            "XY-projected triangles. Holes are shown where the part is hollow."
        )
    else:
        # Large mesh: convex hull is fast and good enough for footprint bounds
        footprint = _convex_hull_footprint(verts[:, :2])
        note = (
            f"Mesh has {n_tris:,} triangles — convex-hull approximation used for "
            "speed. Export a simplified STL for the exact projection."
        )

    if footprint is not None and not footprint.is_empty:
        footprint_area = float(footprint.area)
        # Simplify polygon outline for SVG (tolerance: 0.1% of bbox diagonal)
        tol = max(0.01, math.sqrt(bbox_x ** 2 + bbox_y ** 2) * 0.001)
        simplified = footprint.simplify(tol, preserve_topology=True)
        poly_data = _shapely_to_polygon_data(simplified, x_min, y_min)
    else:
        footprint_area = bbox_x * bbox_y
        poly_data = None

    return {
        "bbox_x":        round(bbox_x, 3),
        "bbox_y":        round(bbox_y, 3),
        "height_z":      round(height_z, 3),
        "bbox_area":     round(bbox_x * bbox_y, 2),
        "footprint_area": round(footprint_area, 2),
        "footprint_polygon": poly_data,
        "source_note":   note,
        "vertex_count":  int(len(verts)),
    }


# ---------------------------------------------------------------------------
# STEP analyser
# ---------------------------------------------------------------------------

def analyze_step(filepath: str) -> dict:
    """
    Extract CARTESIAN_POINT data from an ASCII STEP file (AP203/AP214/AP242).
    Computes convex hull of the projected vertex cloud — best possible without
    full BREP tessellation. For maximum accuracy, upload an STL export.
    """
    try:
        with open(filepath, "r", errors="ignore") as fh:
            content = fh.read()
    except Exception as exc:
        raise ValueError(f"Cannot read STEP file: {exc}") from exc

    if "BINARY" in content[:200].upper():
        raise ValueError(
            "Binary STEP files are not supported. Please export as ASCII STEP."
        )

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
    x_min, y_min, z_min = verts.min(axis=0)
    x_max, y_max, z_max = verts.max(axis=0)
    bbox_x   = float(x_max - x_min)
    bbox_y   = float(y_max - y_min)
    height_z = float(z_max - z_min)

    footprint = _convex_hull_footprint(verts[:, :2])

    if footprint is not None and not footprint.is_empty:
        footprint_area = float(footprint.area)
        poly_data = _shapely_to_polygon_data(footprint, x_min, y_min)
    else:
        footprint_area = bbox_x * bbox_y
        poly_data = None

    note = (
        "STEP file: convex hull of extracted vertex cloud. "
        "Triangle-level projection is not available without full BREP tessellation — "
        "upload an STL for the exact footprint."
    )
    return {
        "bbox_x":        round(bbox_x, 3),
        "bbox_y":        round(bbox_y, 3),
        "height_z":      round(height_z, 3),
        "bbox_area":     round(bbox_x * bbox_y, 2),
        "footprint_area": round(footprint_area, 2),
        "footprint_polygon": poly_data,
        "source_note":   note,
        "vertex_count":  int(len(verts)),
    }


# ---------------------------------------------------------------------------
# Packing calculator
# ---------------------------------------------------------------------------

def packing_estimate(bbox_x: float, bbox_y: float,
                     bed_x: float, bed_y: float, gap: float) -> dict:
    """Rectangular grid packing using bounding-box dimensions + gap."""
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
