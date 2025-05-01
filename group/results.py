# results.py

# ──────────────────────────────────────────────────────────────────────────────
# 0. Force non-interactive matplotlib backend
# ──────────────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')

# ──────────────────────────────────────────────────────────────────────────────
# 1. Imports & Configuration
# ──────────────────────────────────────────────────────────────────────────────
import os, io, base64, json, re, subprocess, threading
from flask import Flask, redirect, url_for, render_template_string, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import DictLoader, ChoiceLoader

app = Flask(__name__)

RESULTS_DIR   = "results"
TICKERS       = ["AMZN", "MSFT"]
CSV_FILES     = {t: f"{t}.csv" for t in TICKERS}
CAT_JSON      = "categorized_transactions_output.json"
RAW_TXT       = "raw_model_responses.txt"
ANOMALY_JSON  = "anomaly_results.json"
MARKET_CSV    = "corporate_market_data.csv"

# Track background job status
job_status = {
    "categorization": False,
    "anomaly": False,
    "trend": False
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. Data Loading Functions
# ──────────────────────────────────────────────────────────────────────────────
def load_categorization_data():
    with open(CAT_JSON) as f:
        transactions = json.load(f)
    models = [p["model"] for p in transactions[0]["predictions"]] if transactions else []
    raw_responses = {}
    txt = open(RAW_TXT).read()
    blocks = re.split(r"^=== Transaction ID:\s*", txt, flags=re.MULTILINE)
    for block in blocks:
        if not block.strip(): continue
        lines = block.splitlines()
        m = re.match(r"(\d+)", lines[0].strip())
        if not m: continue
        txn_id = int(m.group(1))
        model = lines[1].split("Model:")[1].strip()
        resp_lines = []
        for line in lines[3:]:
            if line.startswith("----"): break
            resp_lines.append(line)
        raw_responses.setdefault(txn_id, {})[model] = "\n".join(resp_lines).strip()
    return transactions, models, raw_responses

def load_anomaly_results():
    with open(ANOMALY_JSON) as f:
        data = json.load(f)
    return data.get("model_results", [])

def load_market_data():
    df = pd.read_csv(MARKET_CSV, parse_dates=["Date"]).tail(30)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return list(df.columns), df.to_dict(orient="records")

def load_trend_results():
    data = {}
    for fname in os.listdir(RESULTS_DIR):
        if not fname.startswith("trend_") or not fname.endswith(".json"): continue
        if fname.endswith("_raw.json"):
            rtype, core = "raw", fname[len("trend_"):-len("_raw.json")]
        elif fname.endswith("_numeric.json"):
            rtype, core = "numeric", fname[len("trend_"):-len("_numeric.json")]
        elif fname.endswith("_recommendation.json"):
            rtype, core = "recommendation", fname[len("trend_"):-len("_recommendation.json")]
        else:
            continue
        ticker, model_key = core.split("_", 1)
        model = model_key.replace("_", ":", 1)
        rec = json.load(open(os.path.join(RESULTS_DIR, fname)))
        data.setdefault(ticker, {}).setdefault(rtype, {})[model] = rec
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 3. Inline HTML Templates
# ──────────────────────────────────────────────────────────────────────────────
base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Comparison Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css"
        rel="stylesheet">
  <link href="https://cdn.jsdelivr.net/npm/simple-datatables@latest/dist/style.css"
        rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/simple-datatables@latest"></script>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    pre { white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }
    .disabled { pointer-events: none; opacity: 0.6; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light">
  <div class="container-fluid">
    <a class="navbar-brand nav-link" href="{{ url_for('landing') }}">AI Dashboard</a>
    <ul class="navbar-nav me-auto">
      <li class="nav-item"><a class="nav-link" href="{{ url_for('landing') }}">Home</a></li>
      <li class="nav-item"><a class="nav-link" href="{{ url_for('categorization') }}">Categorization</a></li>
      <li class="nav-item"><a class="nav-link" href="{{ url_for('anomaly') }}">Anomaly Detection</a></li>
      <li class="nav-item"><a class="nav-link" href="{{ url_for('trend') }}">Trend Analysis</a></li>
    </ul>
  </div>
</nav>
<div class="container mt-4">{% block content %}{% endblock %}</div>

<script>
let regenerating = false;
function disableNav() {
  document.querySelectorAll('a.nav-link').forEach(el => el.classList.add('disabled'));
}
function enableNav() {
  document.querySelectorAll('a.nav-link').forEach(el => el.classList.remove('disabled'));
}

document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('.regenerate-btn').forEach(function(btn){
    btn.addEventListener('click', function(e){
      e.preventDefault();
      if (regenerating) return;
      regenerating = true;
      disableNav();
      const comp = btn.dataset.component;
      btn.disabled = true;
      document.getElementById('status-'+comp).innerHTML =
        '<span class="spinner-border spinner-border-sm"></span> Running...';

      fetch('/regenerate/'+comp, {method:'POST'})
        .then(r=>r.json())
        .then(_=>{
          const poll = setInterval(()=>{
            fetch('/status/'+comp).then(r=>r.json()).then(s=>{
              if(!s.running){
                clearInterval(poll);
                regenerating = false;
                enableNav();
                location.reload();
              }
            });
          }, 2000);
        });
    });
  });
});
</script>

</body>
</html>
"""

landing_template = """
{% extends "base" %}
{% block content %}
  <h1>Comparing AI Models for Financial Task Automation</h1>
  <p>This research project develops an AI-powered tool to automate repetitive financial tasks and compare AI models’ effectiveness across three components:</p>
  <ul>
    <li><strong>Transaction Categorization:</strong> Classify transactions and compare model outputs.</li>
    <li><strong>Anomaly Detection:</strong> Flag unusual or suspicious transactions.</li>
    <li><strong>Trend Analysis:</strong> Analyze time series, generate metrics, summaries, and recommendations.</li>
  </ul>
  <p>Use the tabs above to navigate each component’s results:</p>
  <ul>
    <li><strong>Home:</strong> Overview of project goals.</li>
    <li><strong>Categorization:</strong> View categorized transactions and raw AI responses.</li>
    <li><strong>Anomaly Detection:</strong> Explore model analyses and market data.</li>
    <li><strong>Trend Analysis:</strong> View price charts, metrics, LLM summaries, and recommendations.</li>
  </ul>
{% endblock %}
"""

categorization_template = """
{% extends "base" %}
{% block content %}
<h2>Component 1: Transaction Categorization</h2>
<form class="mb-2 d-inline">
  <button type="button" class="btn btn-sm btn-outline-primary regenerate-btn" data-component="categorization">
    Regenerate
  </button>
  <span id="status-categorization"></span>
</form>

<p><strong>Total transactions:</strong> {{ transactions|length }}</p>
<table class="table table-striped">
  <thead>
    <tr>
      <th>ID</th><th>Description</th><th>Amount</th><th>Type</th>
      {% for model in models %}<th>{{ model }}<br><small>(runtime)</small></th>{% endfor %}
    </tr>
  </thead>
  <tbody>
    {% for txn in transactions %}
    <tr>
      <td>{{ txn.transaction_id }}</td>
      <td>{{ txn.description }}</td>
      <td>{{ txn.amount }}</td>
      <td>{{ txn.type }}</td>
      {% for pred in txn.predictions %}
      <td>
        {{ pred.predicted_category }}<br>
        <small>{{ pred.runtime_sec|round(2) }} s</small>
      </td>
      {% endfor %}
    </tr>
    {% endfor %}
  </tbody>
</table>

<h4>Raw Model Responses</h4>
<div class="accordion mb-4" id="txnAccordion">
  {% for txn in transactions %}
  <div class="accordion-item">
    <h2 class="accordion-header" id="heading-txn-{{ txn.transaction_id }}">
      <button class="accordion-button collapsed" type="button"
              data-bs-toggle="collapse"
              data-bs-target="#collapse-txn-{{ txn.transaction_id }}">
        Transaction {{ txn.transaction_id }} Responses
      </button>
    </h2>
    <div id="collapse-txn-{{ txn.transaction_id }}" class="accordion-collapse collapse">
      <div class="accordion-body">
        {% for model, response in raw_responses[txn.transaction_id].items() %}
        <h5>{{ model }}</h5>
        <pre>{{ response }}</pre>
        {% endfor %}
      </div>
    </div>
  </div>
  {% endfor %}
</div>

<h4>Model Runtime Comparison</h4>
<div id="runtime-chart" style="height:400px;"></div>
<script>
  Plotly.newPlot('runtime-chart', {{ runtime_chart|safe }});
</script>
{% endblock %}
"""

anomaly_template = """
{% extends "base" %}
{% block content %}
<h2>Component 2: Anomaly Detection</h2>
<form class="mb-2 d-inline">
  <button type="button" class="btn btn-sm btn-outline-primary regenerate-btn" data-component="anomaly">
    Regenerate
  </button>
  <span id="status-anomaly"></span>
</form>

<h4>Prompt Used</h4>
<pre>{{ prompt_text }}</pre>

<h4>Model Analyses</h4>
<div class="accordion mb-4" id="anomalyAccordion">
  {% for rec in anomaly_results %}
  {% set id_safe = rec.model.replace(':','_') %}
  <div class="accordion-item">
    <h2 class="accordion-header" id="heading-{{ id_safe }}">
      <button class="accordion-button collapsed" type="button"
              data-bs-toggle="collapse"
              data-bs-target="#collapse-{{ id_safe }}">
        {{ rec.model }}
      </button>
    </h2>
    <div id="collapse-{{ id_safe }}" class="accordion-collapse collapse">
      <div class="accordion-body"><pre>{{ rec.raw_output }}</pre></div>
    </div>
  </div>
  {% endfor %}
</div>

<h4>Corporate Market Data (Last 30 Rows)</h4>
<table id="market-table" class="table table-striped">
  <thead>
    <tr>{% for col in market_columns %}<th>{{ col }}</th>{% endfor %}</tr>
  </thead>
  <tbody>
    {% for row in market_rows %}
    <tr>{% for col in market_columns %}<td>{{ row[col] }}</td>{% endfor %}</tr>
    {% endfor %}
  </tbody>
</table>
<script>
  new simpleDatatables.DataTable("#market-table");
</script>
{% endblock %}
"""

trend_template = """
{% extends "base" %}
{% block content %}
<h2>Component 3: Trend Analysis</h2>
<form class="mb-2 d-inline">
  <button type="button" class="btn btn-sm btn-outline-primary regenerate-btn" data-component="trend">
    Regenerate
  </button>
  <span id="status-trend"></span>
</form>

{% for ticker, results in trend_data.items() %}
  <hr><h3>{{ ticker }}</h3>
  <img src="data:image/png;base64,{{ chart_images[ticker] }}" class="img-fluid mb-4"/>

  <h4>Metrics Supplied to LLM</h4>
  <table class="table table-bordered mb-4">
    <thead><tr>
      <th>Model</th><th>Slope/day</th><th>30-day MA</th><th>Annualized Vol (%)</th>
    </tr></thead>
    <tbody>
      {% for model, rec in results.numeric.items() %}
      <tr>
        <td>{{ model }}</td>
        <td>{{ rec.metrics.slope_per_day|round(6) }}</td>
        <td>{{ rec.metrics.last_30d_ma|round(2) }}</td>
        <td>{{ (rec.metrics.annualized_volatility*100)|round(2) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h4>LLM Raw Summaries</h4>
  <div class="accordion mb-4" id="rawAccordion-{{ ticker }}">
    {% for model, rec in results.raw.items() %}
    {% set id_safe = model.replace(':','_') %}
    <div class="accordion-item">
      <h2 class="accordion-header" id="rawHeading-{{ ticker }}-{{ id_safe }}">
        <button class="accordion-button collapsed" type="button"
                data-bs-toggle="collapse"
                data-bs-target="#rawCollapse-{{ ticker }}-{{ id_safe }}">
          {{ model }}
        </button>
      </h2>
      <div id="rawCollapse-{{ ticker }}-{{ id_safe }}" class="accordion-collapse collapse">
        <div class="accordion-body"><pre>{{ rec.summary }}</pre></div>
      </div>
    </div>
    {% endfor %}
  </div>

  <h4>LLM Numeric Interpretations</h4>
  <div class="accordion mb-4" id="numAccordion-{{ ticker }}">
    {% for model, rec in results.numeric.items() %}
    {% set id_safe = model.replace(':','_') %}
    <div class="accordion-item">
      <h2 class="accordion-header" id="numHeading-{{ ticker }}-{{ id_safe }}">
        <button class="accordion-button collapsed" type="button"
                data-bs-toggle="collapse"
                data-bs-target="#numCollapse-{{ ticker }}-{{ id_safe }}">
          {{ model }}
        </button>
      </h2>
      <div id="numCollapse-{{ ticker }}-{{ id_safe }}" class="accordion-collapse collapse">
        <div class="accordion-body"><pre>{{ rec.interpretation }}</pre></div>
      </div>
    </div>
    {% endfor %}
  </div>

  <h4>LLM Investment Recommendations</h4>
  <div class="accordion mb-4" id="recAccordion-{{ ticker }}">
    {% for model, rec in results.recommendation.items() %}
    {% set id_safe = model.replace(':','_') %}
    <div class="accordion-item">
      <h2 class="accordion-header" id="recHeading-{{ ticker }}-{{ id_safe }}">
        <button class="accordion-button collapsed" type="button"
                data-bs-toggle="collapse"
                data-bs-target="#recCollapse-{{ ticker }}-{{ id_safe }}">
          {{ model }}
        </button>
      </h2>
      <div id="recCollapse-{{ ticker }}-{{ id_safe }}" class="accordion-collapse collapse">
        <div class="accordion-body"><pre>{{ rec.recommendation }}</pre></div>
      </div>
    </div>
    {% endfor %}
  </div>
{% endfor %}
{% endblock %}
"""

app.jinja_loader = ChoiceLoader([DictLoader({"base": base_template}), app.jinja_loader])


# ──────────────────────────────────────────────────────────────────────────────
# 4. Helpers
# ──────────────────────────────────────────────────────────────────────────────
def plot_to_base64(ticker):
    df = pd.read_csv(CSV_FILES[ticker], parse_dates=["Date"]).sort_values("Date")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], df["Adj Close"], marker="o", linestyle="-")
    ax.set_title(f"{ticker} Adjusted Close — Full History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

def run_component(component):
    job_status[component] = True
    try:
        if component == "categorization":
            subprocess.run(["python", "component1.py"], check=False)
        elif component == "anomaly":
            subprocess.run(["python", "component2.py"], check=False)
        elif component == "trend":
            subprocess.run(["python", "trendanalysis.py"], check=False)
            subprocess.run(["python", "trendrecs.py"], check=False)
    finally:
        job_status[component] = False


# ──────────────────────────────────────────────────────────────────────────────
# 5. Routes
# ──────────────────────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template_string(landing_template)

@app.route("/categorization")
def categorization():
    trans, models, raw_resp = load_categorization_data()
    # runtime stats
    runtimes = {m: [] for m in models}
    for txn in trans:
        for p in txn["predictions"]:
            runtimes[p["model"]].append(p["runtime_sec"])
    totals = [sum(runtimes[m]) for m in models]
    avgs   = [sum(runtimes[m])/len(runtimes[m]) if runtimes[m] else 0 for m in models]
    fig = go.Figure(data=[
        go.Bar(name="Total Runtime (s)",   x=models, y=totals),
        go.Bar(name="Average Runtime (s)", x=models, y=avgs)
    ])
    fig.update_layout(barmode="group", title="Model Runtime: Total vs. Average",
                      xaxis_title="Model", yaxis_title="Seconds")
    runtime_chart = pio.to_json(fig)
    return render_template_string(
        categorization_template,
        transactions=trans,
        models=models,
        raw_responses=raw_resp,
        runtime_chart=runtime_chart
    )

@app.route("/anomaly")
def anomaly():
    prompt_text = (
        "Analyze the following dataset and identify any financial anomalies. "
        "For each anomaly, provide transaction_id, reason, and confidence_score if available."
    )
    anomaly_results = load_anomaly_results()
    cols, rows = load_market_data()
    return render_template_string(
        anomaly_template,
        prompt_text=prompt_text,
        anomaly_results=anomaly_results,
        market_columns=cols,
        market_rows=rows
    )

@app.route("/trend")
def trend():
    data = load_trend_results()
    charts = {t: plot_to_base64(t) for t in TICKERS}
    return render_template_string(
        trend_template,
        trend_data=data,
        chart_images=charts
    )

@app.route("/regenerate/<component>", methods=["POST"])
def regenerate(component):
    if component not in job_status:
        return jsonify({"error":"Unknown component"}), 404
    if not job_status[component]:
        threading.Thread(target=run_component, args=(component,), daemon=True).start()
    return jsonify({"running": True})

@app.route("/status/<component>")
def status(component):
    return jsonify({"running": job_status.get(component, False)})

# ──────────────────────────────────────────────────────────────────────────────
# 6. App Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
