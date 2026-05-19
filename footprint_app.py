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

def analyze_stl(filepath: str) -> dict:
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

    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    bbox_x = float(maxs[0] - mins[0])
    bbox_y = float(maxs[1] - mins[1])
    height = float(maxs[2] - mins[2])
    x_min, y_min = float(mins[0]), float(mins[1])

    try:
        footprint = trimesh_projected(mesh, normal=[0.0, 0.0, 1.0])
    except Exception:
        footprint = None

    if footprint is None or footprint.is_empty:
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
# STEP analyser
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
# Multi-part packing (shelf algorithm)
# ---------------------------------------------------------------------------

def multi_packing_estimate(parts_info: list, bed_x: float, bed_y: float, gap: float) -> dict:
    """
    Greedy shelf packing for a mixed list of parts with quantities.
    parts_info: list of {"bbox_x", "bbox_y", "qty"} ordered by part index.
    """
    items = []
    for i, p in enumerate(parts_info):
        for _ in range(int(p["qty"])):
            items.append({"w": float(p["bbox_x"]), "h": float(p["bbox_y"]), "idx": i})
    # Sort tallest first so each shelf's height is set by its first (tallest) item
    items.sort(key=lambda x: x["h"], reverse=True)

    placements = []
    cur_x = 0.0
    cur_y = 0.0
    shelf_h = 0.0

    for item in items:
        w, h = item["w"], item["h"]
        if w > bed_x + 1e-9 or h > bed_y + 1e-9:
            continue  # single item too large for bed
        if cur_x + w <= bed_x + 1e-9:
            placements.append({"x": round(cur_x, 3), "y": round(cur_y, 3),
                               "w": round(w, 3), "h": round(h, 3), "idx": item["idx"]})
            if shelf_h == 0:
                shelf_h = h
            cur_x += w + gap
        else:
            cur_y += shelf_h + gap
            cur_x = 0.0
            shelf_h = h
            if cur_y + h > bed_y + 1e-9:
                break
            placements.append({"x": round(cur_x, 3), "y": round(cur_y, 3),
                               "w": round(w, 3), "h": round(h, 3), "idx": item["idx"]})
            cur_x = w + gap

    total_needed = sum(int(p["qty"]) for p in parts_info)
    placed_count = len(placements)
    bed_area = bed_x * bed_y
    placed_area = sum(pl["w"] * pl["h"] for pl in placements)
    fill_pct = round(placed_area / bed_area * 100, 1) if bed_area > 0 else 0.0

    return {
        "placements":   placements,
        "placed_count": placed_count,
        "total_needed": total_needed,
        "all_fit":      placed_count >= total_needed,
        "fill_pct":     fill_pct,
    }


def multi_bed_assessment(pack_result: dict) -> dict:
    placed = pack_result["placed_count"]
    needed = pack_result["total_needed"]
    fill   = pack_result["fill_pct"]

    if not pack_result["all_fit"]:
        level = "overflow"
        msg   = f"Only {placed} of {needed} parts fit. Reduce quantities or use a larger bed."
        color = "red"
    elif fill >= 80:
        level, color = "full", "red"
        msg = "Bed is essentially full with all specified parts."
    elif fill >= 50:
        level, color = "crowded", "amber"
        msg = "More than half the bed is occupied. Limited room to add more parts."
    elif fill >= 20:
        level, color = "moderate", "blue"
        msg = "All parts fit with reasonable space available."
    else:
        level, color = "spacious", "green"
        msg = "All parts fit with plenty of bed space remaining."

    return {"level": level, "message": msg, "color": color,
            "all_fit": pack_result["all_fit"], "fill_pct": fill}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("footprint.html")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    bed_x = float(request.form.get("bed_x", 200))
    bed_y = float(request.form.get("bed_y", 200))
    gap   = float(request.form.get("gap",   GAP_DEFAULT))

    file_entries = []
    i = 0
    while f"file_{i}" in request.files:
        f = request.files[f"file_{i}"]
        qty = max(1, int(request.form.get(f"qty_{i}", 1) or 1))
        if f.filename:
            file_entries.append((f, qty))
        i += 1

    if not file_entries:
        return jsonify({"error": "No files uploaded."}), 400

    parts = []
    for f, qty in file_entries:
        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"Unsupported format '{ext}' in '{f.filename}'. "
                                      "Upload .stl, .step, or .stp."}), 400

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            geo = analyze_stl(tmp_path) if ext == ".stl" else analyze_step(tmp_path)
        except Exception as exc:
            return jsonify({"error": f"{f.filename}: {exc}"}), 422
        finally:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass

        parts.append({"filename": f.filename, "qty": qty, "geometry": geo})

    pack = multi_packing_estimate(
        [{"bbox_x": p["geometry"]["bbox_x"], "bbox_y": p["geometry"]["bbox_y"],
          "qty": p["qty"]} for p in parts],
        bed_x, bed_y, gap,
    )
    assessment = multi_bed_assessment(pack)

    return jsonify({
        "parts":      parts,
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
