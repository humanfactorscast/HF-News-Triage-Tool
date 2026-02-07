from __future__ import annotations

from flask import Flask, render_template, request

from app.hf_scoring import parse_headlines, score_headlines

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    headlines_text = ""
    results = []
    score_threshold = 5
    active_domains: list[str] = []

    if request.method == "POST":
        headlines_text = request.form.get("headlines", "")
        score_threshold = int(request.form.get("score_threshold", 5))
        active_domains = request.form.getlist("domains")
        headlines = parse_headlines(headlines_text)
        results = score_headlines(
            headlines=headlines,
            active_domains=active_domains,
            score_threshold=score_threshold,
        )

    return render_template(
        "index.html",
        headlines_text=headlines_text,
        results=results,
        score_threshold=score_threshold,
        active_domains=active_domains,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
