import os
import uuid
import threading
import traceback

from flask import Flask, render_template, request, jsonify, send_from_directory

from pipeline import run_pipeline, OUTPUT_DIR

app = Flask(__name__)

JOBS = {}  # job_id -> {"status": str, "log": [str], "results": [...], "error": str|None}


def process_job(job_id, url, num_clips, clip_len):
    JOBS[job_id] = {"status": "running", "log": [], "results": [], "error": None}

    def progress(msg):
        JOBS[job_id]["log"].append(msg)

    try:
        results = run_pipeline(url, job_id, num_clips=num_clips, clip_len=clip_len, progress_cb=progress)
        JOBS[job_id]["results"] = results
        JOBS[job_id]["status"] = "done"
    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = f"{e}\n{traceback.format_exc()}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def start():
    data = request.get_json()
    url = data.get("url", "").strip()
    num_clips = int(data.get("num_clips", 5))
    clip_len = int(data.get("clip_len", 45))

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    job_id = uuid.uuid4().hex[:10]
    thread = threading.Thread(target=process_job, args=(job_id, url, num_clips, clip_len), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


@app.route("/downloads/<path:filename>")
def download_clip(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
