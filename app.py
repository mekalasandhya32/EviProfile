"""Local web interface (no database, no external services).

    python run_pipeline.py      # once: trains and saves models/eviprofile.joblib
    python app.py               # then open http://127.0.0.1:5000
"""
import tempfile
from pathlib import Path

from flask import Flask, render_template_string, request

from resume_intel import config as C
from resume_intel.pipeline import EviProfilePipeline, render_html_card, CARD_CSS, job_from_text

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024          # 5 MB upload limit
_PIPE = None

PAGE = """<!doctype html><title>EviProfile</title><style>""" + CARD_CSS + """
form{max-width:900px;margin:0 auto 20px;background:#fff;padding:18px 24px;border-radius:10px;box-shadow:0 2px 8px #0002}
input,textarea{width:100%;box-sizing:border-box;margin:4px 0 10px;padding:6px;font:inherit}h1{text-align:center;color:#1b3a57}
button{background:#0B6E4F;color:#fff;border:0;padding:9px 22px;border-radius:6px;font-size:15px;cursor:pointer}.err{color:#c0392b;text-align:center}</style>
<h1>EviProfile: evidence-grounded candidate profiling</h1>
<form method=post enctype=multipart/form-data>
<label><b>Resume file</b> (.txt, .docx or .pdf)</label><input type=file name=resume accept=".txt,.docx,.pdf">
<label><b>Job requirements (optional)</b></label>
<input name=required placeholder="Required skills, e.g. Python, Machine Learning, TensorFlow, SQL" value="{{ req }}">
<input name=preferred placeholder="Preferred skills, e.g. Tableau, NLP" value="{{ pref }}">
<input name=min_years type=number min=0 max=40 placeholder="Minimum years of experience" value="{{ miny }}">
<button type=submit>Analyse resume</button></form>
{% if error %}<p class=err>{{ error }}</p>{% endif %}{{ card|safe }}"""


def pipe() -> EviProfilePipeline:
    global _PIPE
    if _PIPE is None:
        path = C.MODEL_DIR / "eviprofile.joblib"
        if not path.exists():
            raise RuntimeError("Model not found. Run `python run_pipeline.py` first.")
        _PIPE = EviProfilePipeline.load(path)
    return _PIPE


@app.route("/", methods=["GET", "POST"])
def index():
    card, error = "", ""
    req = request.form.get("required", "")
    pref = request.form.get("preferred", "")
    miny = request.form.get("min_years", "")
    if request.method == "POST":
        f = request.files.get("resume")
        if not f or not f.filename:
            error = "Please choose a resume file."
        else:
            ext = Path(f.filename).suffix.lower()
            if ext not in (".txt", ".docx", ".pdf"):
                error = "Unsupported file type. Use .txt, .docx or .pdf."
            else:
                try:
                    with tempfile.TemporaryDirectory() as d:
                        p = Path(d) / f"upload{ext}"                    # never trust the client's file name
                        f.save(p)
                        job = job_from_text(req, pref, int(miny or 0)) if req.strip() else None
                        prof = pipe().analyze_file(p, job)
                    card = render_html_card(prof)
                except Exception as e:                                   # show a friendly message, not a stack trace
                    error = f"Could not analyse this file: {e}"
    return render_template_string(PAGE, card=card, error=error, req=req, pref=pref, miny=miny)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
