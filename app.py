from flask import Flask, render_template, request, jsonify
import sqlite3
import os
from datetime import datetime, timedelta

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "manufacturing.db")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS machines (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            status      TEXT    NOT NULL DEFAULT 'idle',
            build_x     REAL    NOT NULL DEFAULT 250,
            build_y     REAL    NOT NULL DEFAULT 250,
            build_z     REAL    NOT NULL DEFAULT 300,
            notes       TEXT,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS orders (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            client_name TEXT    NOT NULL,
            reference   TEXT,
            status      TEXT    NOT NULL DEFAULT 'pending',
            due_date    TEXT,
            notes       TEXT,
            created_at  TEXT    DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS parts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id    INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
            name        TEXT    NOT NULL,
            quantity    INTEGER NOT NULL DEFAULT 1,
            size_x      REAL,
            size_y      REAL,
            size_z      REAL
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id       INTEGER NOT NULL REFERENCES machines(id),
            order_id         INTEGER NOT NULL REFERENCES orders(id),
            status           TEXT    NOT NULL DEFAULT 'queued',
            estimated_hours  REAL    DEFAULT 0,
            started_at       TEXT,
            completed_at     TEXT,
            notes            TEXT,
            created_at       TEXT    DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def machine_queue_hours(conn, machine_id):
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_hours), 0) FROM jobs "
        "WHERE machine_id = ? AND status IN ('queued', 'printing')",
        (machine_id,),
    ).fetchone()
    return float(row[0])


def machine_util_hours(conn, machine_id, days=30):
    cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    row = conn.execute(
        "SELECT COALESCE(SUM(estimated_hours), 0) FROM jobs "
        "WHERE machine_id = ? AND status = 'completed' AND completed_at >= ?",
        (machine_id, cutoff),
    ).fetchone()
    return float(row[0])


def machine_queued_jobs(conn, machine_id):
    rows = conn.execute(
        """SELECT j.id, j.status, j.estimated_hours, j.created_at,
                  o.client_name, o.reference
           FROM jobs j JOIN orders o ON j.order_id = o.id
           WHERE j.machine_id = ? AND j.status IN ('queued','printing')
           ORDER BY j.created_at""",
        (machine_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def enrich_machine(conn, m):
    m = dict(m)
    m["queue_hours"] = round(machine_queue_hours(conn, m["id"]), 1)
    m["utilization_hours"] = round(machine_util_hours(conn, m["id"]), 1)
    m["queued_jobs"] = machine_queued_jobs(conn, m["id"])
    return m


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

def recommend(conn, order_id):
    order = dict(conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone())
    parts = [dict(r) for r in conn.execute("SELECT * FROM parts WHERE order_id = ?", (order_id,)).fetchall()]

    # Footprint area of all parts (for batch check)
    total_area = sum(
        (p.get("size_x") or 0) * (p.get("size_y") or 0) * (p.get("quantity") or 1)
        for p in parts
    )

    machines = [dict(r) for r in conn.execute("SELECT * FROM machines ORDER BY name").fetchall()]
    results = []

    for m in machines:
        mid = m["id"]
        unavailable = m["status"] in ("maintenance", "offline")

        if unavailable:
            results.append({
                "machine": m,
                "score": None,
                "queue_hours": None,
                "utilization_hours": None,
                "batch_opportunity": False,
                "batch_job": None,
                "available": False,
                "reasons": [],
                "warnings": [f"Machine is currently {m['status']}"],
            })
            continue

        q_hours = machine_queue_hours(conn, mid)
        u_hours = machine_util_hours(conn, mid)
        queued = machine_queued_jobs(conn, mid)

        # Batch detection: new parts fit on the plate with an existing queued job?
        batch = False
        batch_job = None
        if queued and total_area > 0:
            plate_area = m["build_x"] * m["build_y"]
            if plate_area > 0 and total_area < 0.40 * plate_area:
                batch = True
                batch_job = queued[-1]

        # Scoring — lower is better
        score = q_hours
        score += u_hours * 0.08           # mild load-balance penalty
        if batch:
            score -= 5.0                  # strong incentive to combine builds
        if m["status"] == "idle":
            score -= 1.5                  # idle machines get a small boost

        reasons = []
        if m["status"] == "idle":
            reasons.append("Machine is currently idle — can start right away")
        if q_hours == 0:
            reasons.append("No jobs in queue")
        else:
            reasons.append(f"{q_hours:.1f} h of work already queued")
        if u_hours > 0:
            reasons.append(f"{u_hours:.0f} h used in the last 30 days")
        if batch and batch_job:
            label = batch_job.get("reference") or batch_job.get("client_name") or f"job #{batch_job['id']}"
            reasons.append(f"Parts may fit on the same build plate as '{label}' — saves a full setup")

        results.append({
            "machine": m,
            "score": round(score, 2),
            "queue_hours": round(q_hours, 1),
            "utilization_hours": round(u_hours, 1),
            "batch_opportunity": batch,
            "batch_job": batch_job,
            "available": True,
            "reasons": reasons,
            "warnings": [],
        })

    available = sorted([r for r in results if r["available"]], key=lambda x: x["score"])
    unavailable = [r for r in results if not r["available"]]
    return available + unavailable


# ---------------------------------------------------------------------------
# Routes — pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Routes — machines
# ---------------------------------------------------------------------------

@app.route("/api/machines", methods=["GET", "POST"])
def api_machines():
    conn = get_db()
    try:
        if request.method == "GET":
            rows = conn.execute("SELECT * FROM machines ORDER BY name").fetchall()
            return jsonify([enrich_machine(conn, r) for r in rows])

        data = request.get_json()
        conn.execute(
            "INSERT INTO machines (name, status, build_x, build_y, build_z, notes) VALUES (?,?,?,?,?,?)",
            (data["name"], data.get("status", "idle"),
             data.get("build_x", 250), data.get("build_y", 250), data.get("build_z", 300),
             data.get("notes", "")),
        )
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()


@app.route("/api/machines/<int:mid>", methods=["PUT", "DELETE"])
def api_machine(mid):
    conn = get_db()
    try:
        if request.method == "DELETE":
            conn.execute("DELETE FROM machines WHERE id = ?", (mid,))
            conn.commit()
            return jsonify({"success": True})

        data = request.get_json()
        allowed = ["name", "status", "build_x", "build_y", "build_z", "notes"]
        fields = {k: data[k] for k in allowed if k in data}
        if fields:
            sql = ", ".join(f"{k} = ?" for k in fields)
            conn.execute(f"UPDATE machines SET {sql} WHERE id = ?", list(fields.values()) + [mid])
            conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes — orders
# ---------------------------------------------------------------------------

@app.route("/api/orders", methods=["GET", "POST"])
def api_orders():
    conn = get_db()
    try:
        if request.method == "GET":
            rows = conn.execute("SELECT * FROM orders ORDER BY created_at DESC").fetchall()
            result = []
            for r in rows:
                o = dict(r)
                o["parts"] = [dict(p) for p in conn.execute(
                    "SELECT * FROM parts WHERE order_id = ?", (o["id"],)).fetchall()]
                job = conn.execute(
                    """SELECT j.*, m.name AS machine_name FROM jobs j
                       JOIN machines m ON j.machine_id = m.id
                       WHERE j.order_id = ? ORDER BY j.created_at DESC LIMIT 1""",
                    (o["id"],),
                ).fetchone()
                o["job"] = dict(job) if job else None
                result.append(o)
            return jsonify(result)

        data = request.get_json()
        cur = conn.execute(
            "INSERT INTO orders (client_name, reference, due_date, notes) VALUES (?,?,?,?)",
            (data["client_name"], data.get("reference", ""),
             data.get("due_date", ""), data.get("notes", "")),
        )
        order_id = cur.lastrowid
        for p in data.get("parts", []):
            conn.execute(
                "INSERT INTO parts (order_id, name, quantity, size_x, size_y, size_z) VALUES (?,?,?,?,?,?)",
                (order_id, p["name"], p.get("quantity", 1),
                 p.get("size_x"), p.get("size_y"), p.get("size_z")),
            )
        conn.commit()
        return jsonify({"id": order_id, "success": True})
    finally:
        conn.close()


@app.route("/api/orders/<int:oid>", methods=["GET", "DELETE"])
def api_order(oid):
    conn = get_db()
    try:
        if request.method == "DELETE":
            conn.execute("DELETE FROM orders WHERE id = ?", (oid,))
            conn.commit()
            return jsonify({"success": True})
        o = dict(conn.execute("SELECT * FROM orders WHERE id = ?", (oid,)).fetchone())
        o["parts"] = [dict(p) for p in conn.execute(
            "SELECT * FROM parts WHERE order_id = ?", (oid,)).fetchall()]
        return jsonify(o)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes — recommendation & assignment
# ---------------------------------------------------------------------------

@app.route("/api/recommend/<int:order_id>")
def api_recommend(order_id):
    conn = get_db()
    try:
        return jsonify(recommend(conn, order_id))
    finally:
        conn.close()


@app.route("/api/assign", methods=["POST"])
def api_assign():
    conn = get_db()
    try:
        data = request.get_json()
        machine_id = data["machine_id"]
        order_id = data["order_id"]
        est_hours = float(data.get("estimated_hours") or 0)

        # Guard: order must not already be assigned
        existing = conn.execute(
            "SELECT id FROM jobs WHERE order_id = ? AND status NOT IN ('cancelled','completed')",
            (order_id,),
        ).fetchone()
        if existing:
            return jsonify({"error": "Order is already assigned to a machine"}), 400

        machine = dict(conn.execute("SELECT * FROM machines WHERE id = ?", (machine_id,)).fetchone())

        # If machine is idle, start printing immediately; otherwise queue it
        if machine["status"] == "idle":
            job_status = "printing"
            started_at = datetime.now().isoformat()
            conn.execute("UPDATE machines SET status = 'printing' WHERE id = ?", (machine_id,))
            conn.execute("UPDATE orders SET status = 'printing' WHERE id = ?", (order_id,))
        else:
            job_status = "queued"
            started_at = None
            conn.execute("UPDATE orders SET status = 'scheduled' WHERE id = ?", (order_id,))

        conn.execute(
            "INSERT INTO jobs (machine_id, order_id, status, estimated_hours, started_at, notes) VALUES (?,?,?,?,?,?)",
            (machine_id, order_id, job_status, est_hours, started_at, data.get("notes", "")),
        )
        conn.commit()
        return jsonify({"success": True, "job_status": job_status})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes — jobs
# ---------------------------------------------------------------------------

@app.route("/api/jobs", methods=["GET"])
def api_jobs():
    conn = get_db()
    try:
        rows = conn.execute(
            """SELECT j.*, m.name AS machine_name, o.client_name, o.reference
               FROM jobs j
               JOIN machines m ON j.machine_id = m.id
               JOIN orders o   ON j.order_id   = o.id
               ORDER BY j.created_at DESC"""
        ).fetchall()
        return jsonify([dict(r) for r in rows])
    finally:
        conn.close()


@app.route("/api/jobs/<int:job_id>", methods=["PUT"])
def api_job(job_id):
    conn = get_db()
    try:
        data = request.get_json()
        new_status = data.get("status")
        if not new_status:
            return jsonify({"error": "status required"}), 400

        job = dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        machine_id = job["machine_id"]
        order_id = job["order_id"]

        updates = {"status": new_status}

        if new_status == "printing":
            updates["started_at"] = datetime.now().isoformat()
            conn.execute("UPDATE orders SET status = 'printing' WHERE id = ?", (order_id,))
            conn.execute("UPDATE machines SET status = 'printing' WHERE id = ?", (machine_id,))

        elif new_status == "completed":
            updates["completed_at"] = datetime.now().isoformat()
            conn.execute("UPDATE orders SET status = 'completed' WHERE id = ?", (order_id,))
            # Check remaining active jobs on this machine
            remaining = conn.execute(
                "SELECT id, order_id FROM jobs WHERE machine_id = ? AND status IN ('queued','printing') AND id != ?",
                (machine_id, job_id),
            ).fetchall()
            if not remaining:
                conn.execute("UPDATE machines SET status = 'idle' WHERE id = ?", (machine_id,))
            else:
                # Auto-start next queued job
                next_job = conn.execute(
                    "SELECT id, order_id FROM jobs WHERE machine_id = ? AND status = 'queued' AND id != ? ORDER BY created_at LIMIT 1",
                    (machine_id, job_id),
                ).fetchone()
                if next_job:
                    conn.execute(
                        "UPDATE jobs SET status = 'printing', started_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), next_job["id"]),
                    )
                    conn.execute("UPDATE orders SET status = 'printing' WHERE id = ?", (next_job["order_id"],))

        elif new_status == "cancelled":
            conn.execute("UPDATE orders SET status = 'pending' WHERE id = ?", (order_id,))
            # If machine has no other active jobs, set idle
            remaining = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE machine_id = ? AND status IN ('queued','printing') AND id != ?",
                (machine_id, job_id),
            ).fetchone()[0]
            if remaining == 0:
                conn.execute("UPDATE machines SET status = 'idle' WHERE id = ?", (machine_id,))

        sql = ", ".join(f"{k} = ?" for k in updates)
        conn.execute(f"UPDATE jobs SET {sql} WHERE id = ?", list(updates.values()) + [job_id])
        conn.commit()
        return jsonify({"success": True})
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes — dashboard summary
# ---------------------------------------------------------------------------

@app.route("/api/dashboard")
def api_dashboard():
    conn = get_db()
    try:
        machine_status = {r["status"]: r["cnt"] for r in conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM machines GROUP BY status").fetchall()}
        order_status = {r["status"]: r["cnt"] for r in conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM orders GROUP BY status").fetchall()}
        active_jobs = [dict(r) for r in conn.execute(
            """SELECT j.*, m.name AS machine_name, o.client_name, o.reference
               FROM jobs j
               JOIN machines m ON j.machine_id = m.id
               JOIN orders o   ON j.order_id   = o.id
               WHERE j.status IN ('queued','printing')
               ORDER BY j.status DESC, j.created_at""",
        ).fetchall()]
        return jsonify({
            "machine_status": machine_status,
            "order_status": order_status,
            "active_jobs": active_jobs,
        })
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Routes — Gantt / schedule
# ---------------------------------------------------------------------------

@app.route("/api/gantt")
def api_gantt():
    conn = get_db()
    try:
        now = datetime.now()
        machines = [dict(r) for r in conn.execute("SELECT * FROM machines ORDER BY name").fetchall()]

        result = []
        for m in machines:
            # Active jobs in execution order: printing first, then queued by created_at
            jobs = [dict(r) for r in conn.execute(
                """SELECT j.*, o.client_name, o.reference
                   FROM jobs j JOIN orders o ON j.order_id = o.id
                   WHERE j.machine_id = ? AND j.status IN ('printing','queued')
                   ORDER BY
                     CASE j.status WHEN 'printing' THEN 0 ELSE 1 END,
                     j.created_at""",
                (m["id"],),
            ).fetchall()]

            # Calculate each job's start/end on the timeline
            cursor = now
            for job in jobs:
                hours = float(job["estimated_hours"] or 0)
                if job["status"] == "printing" and job["started_at"]:
                    start = datetime.fromisoformat(job["started_at"])
                    end = start + timedelta(hours=hours)
                    # cursor advances to whenever this finishes (may be in the past if overdue)
                    cursor = max(end, now)
                else:
                    start = cursor
                    end = cursor + timedelta(hours=hours)
                    cursor = end
                job["gantt_start"] = start.isoformat()
                job["gantt_end"] = end.isoformat()

            m["jobs"] = jobs
            result.append(m)

        # Pending orders (not yet assigned)
        pending = [dict(r) for r in conn.execute(
            """SELECT o.id, o.client_name, o.reference, o.due_date,
                      COUNT(p.id) AS part_count,
                      COALESCE(SUM(p.quantity), 0) AS total_qty
               FROM orders o LEFT JOIN parts p ON p.order_id = o.id
               WHERE o.status = 'pending'
               GROUP BY o.id
               ORDER BY o.created_at DESC""",
        ).fetchall()]

        return jsonify({"machines": result, "pending": pending, "now": now.isoformat()})
    finally:
        conn.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print()
    print("=" * 54)
    print("  LPBF Machine Control System")
    print("  Open your browser at:  http://localhost:5000")
    print("=" * 54)
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
