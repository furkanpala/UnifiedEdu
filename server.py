# server.py
import os
import time
import threading
import logging
from functools import wraps
from typing import Dict, Any

from flask import Flask, request, jsonify

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("server.log"),
    ],
)
logger = logging.getLogger(__name__)

# ── Flask App — created immediately, no heavy imports yet ──────────────────────
app = Flask(__name__)

# ── State ──────────────────────────────────────────────────────────────────────
FULL_MODE    = False   # Will be set to True after lazy init succeeds
MISSING_DEPS = []
INIT_DONE    = False   # Guard so we only init once
INIT_LOCK    = threading.Lock()

global_gnn       = None
aggregation_lock = threading.Lock()
pending_updates: Dict[str, Dict[str, Any]] = {}
current_round    = 0
round_start_time = time.time()

MIN_CLIENTS_TO_AGGREGATE = int(os.environ.get("MIN_CLIENTS", 2))
ROUND_TIMEOUT_SECONDS    = int(os.environ.get("ROUND_TIMEOUT", 300))


# ── Lazy initializer — called once on first real request ───────────────────────

def try_init_ml_stack():
    """
    Attempt to import and initialize the ML stack.
    Called lazily on the first request so the server
    boots instantly and passes Render's health check.
    """
    global FULL_MODE, MISSING_DEPS, INIT_DONE, global_gnn

    with INIT_LOCK:
        if INIT_DONE:
            return  # Already attempted, don't retry
        INIT_DONE = True

        logger.info("Attempting to initialize ML stack (lazy)...")

        try:
            import torch
            from gnn.model import ModelGraphGNN
            from config import GNNConfig

            gnn_cfg    = GNNConfig()
            global_gnn = ModelGraphGNN(gnn_cfg)
            global_gnn.eval()
            FULL_MODE = True
            logger.info("✅ ML stack initialized. Server now in FULL MODE.")

        except ImportError as e:
            MISSING_DEPS.append(str(e))
            logger.warning(f"⚠️  ML stack unavailable: {e}. Running in LIMITED MODE.")

        except Exception as e:
            MISSING_DEPS.append(str(e))
            logger.error(f"❌ ML stack init failed: {e}. Running in LIMITED MODE.")


# ── Before each request: ensure ML stack init has been attempted ───────────────

@app.before_request
def ensure_initialized():
    """Trigger lazy ML init on the very first request."""
    if not INIT_DONE:
        # Run in background thread so the current request
        # is NOT blocked by the heavy import
        t = threading.Thread(target=try_init_ml_stack, daemon=True)
        t.start()


# ── Decorator: require FULL_MODE ───────────────────────────────────────────────

def requires_full_mode(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not FULL_MODE:
            return jsonify({
                "error":   "This endpoint requires the full ML stack.",
                "reason":  "Server is running in limited mode.",
                "missing": MISSING_DEPS,
                "fix":     "pip install torch torch-geometric transformers",
            }), 503
        return f(*args, **kwargs)
    return decorated


# ── Helper functions ───────────────────────────────────────────────────────────

def flatten_params(state_dict: dict) -> list:
    import torch
    return torch.cat([
        v.cpu().float().flatten() for v in state_dict.values()
    ]).tolist()


def unflatten_params(flat: list, reference_state_dict: dict) -> dict:
    import torch
    flat_tensor = torch.tensor(flat, dtype=torch.float32)
    new_state   = {}
    offset      = 0
    for key, ref_val in reference_state_dict.items():
        numel          = ref_val.numel()
        new_state[key] = flat_tensor[offset: offset + numel].reshape(ref_val.shape)
        offset        += numel
    return new_state


def fedavg(updates: Dict[str, Dict]) -> dict:
    import torch
    total_samples = sum(u["num_samples"] for u in updates.values())
    ref_state     = global_gnn.state_dict()
    aggregated    = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in ref_state.items()
    }
    for client_id, update in updates.items():
        weight      = update["num_samples"] / total_samples
        client_dict = unflatten_params(update["parameters"], ref_state)
        for key in aggregated:
            aggregated[key] += weight * client_dict[key]
    logger.info(f"FedAvg | clients: {len(updates)} | samples: {total_samples}")
    return aggregated


def run_aggregation():
    import torch
    global global_gnn, current_round, round_start_time, pending_updates

    with aggregation_lock:
        if len(pending_updates) < MIN_CLIENTS_TO_AGGREGATE:
            return

        logger.info(f"=== Aggregating Round {current_round + 1} ===")
        aggregated_params = fedavg(pending_updates)
        global_gnn.load_state_dict(aggregated_params)

        os.makedirs("checkpoints", exist_ok=True)
        ckpt = f"checkpoints/global_gnn_round_{current_round + 1}.pt"
        torch.save(global_gnn.state_dict(), ckpt)
        logger.info(f"Checkpoint saved → {ckpt}")

        current_round    += 1
        round_start_time  = time.time()
        pending_updates   = {}


def timeout_aggregation_loop():
    while True:
        time.sleep(30)
        if not FULL_MODE:
            continue
        elapsed = time.time() - round_start_time
        with aggregation_lock:
            has_updates = len(pending_updates) > 0
        if has_updates and elapsed > ROUND_TIMEOUT_SECONDS:
            logger.info("Timeout reached. Forcing aggregation.")
            run_aggregation()


threading.Thread(target=timeout_aggregation_loop, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# Routes — always available
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Always responds instantly — no ML deps needed."""
    return jsonify({
        "status":          "ok",
        "full_mode":       FULL_MODE,
        "init_done":       INIT_DONE,
        "missing_deps":    MISSING_DEPS,
        "current_round":   current_round if FULL_MODE else None,
        "pending_clients": list(pending_updates.keys()) if FULL_MODE else [],
        "min_clients":     MIN_CLIENTS_TO_AGGREGATE,
        "message": (
            "Server fully operational."
            if FULL_MODE else
            "Limited mode — ML stack loading or unavailable."
        ),
    }), 200


@app.route("/", methods=["GET"])
def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UnifiedEdu — Aggregation Server</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1117; color: #e2e8f0;
            min-height: 100vh; display: flex; flex-direction: column;
            align-items: center; justify-content: center; padding: 2rem;
        }.container { max-width: 780px; width: 100%; }.badge {
            display: inline-block; background: #1a1f2e; border: 1px solid #2d3748;
            color: #68d391; font-size: 0.72rem; font-weight: 600;
            letter-spacing: 0.08em; text-transform: uppercase;
            padding: 0.3rem 0.85rem; border-radius: 999px; margin-bottom: 1.4rem;
        }
        h1 {
            font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #ffffff 0%, #a0aec0 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
        }.subtitle { color: #718096; font-size: 1rem; margin-bottom: 0.35rem; }.affiliation { color: #4a5568; font-size: 0.85rem; margin-bottom: 2rem; }.affiliation a { color: #63b3ed; text-decoration: none; }.affiliation a:hover { text-decoration: underline; }.status-bar {
            display: flex; align-items: center; gap: 0.6rem;
            background: #0d1f17; border: 1px solid #1a4731;
            border-radius: 10px; padding: 0.75rem 1.2rem;
            margin-bottom: 2rem; font-size: 0.88rem; color: #68d391;
        }.dot {
            width: 8px; height: 8px; background: #48bb78;
            border-radius: 50%; animation: pulse 2s infinite;
        }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }.section-title {
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: #4a5568; margin-bottom: 0.85rem;
        }.description {
            background: #1a1f2e; border: 1px solid #2d3748; border-radius: 12px;
            padding: 1.4rem 1.6rem; margin-bottom: 1.5rem;
            font-size: 0.93rem; line-height: 1.75; color: #a0aec0;
        }.description strong { color: #e2e8f0; }.endpoints {
            background: #1a1f2e; border: 1px solid #2d3748;
            border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem;
        }.endpoint-header {
            padding: 0.85rem 1.4rem; border-bottom: 1px solid #2d3748;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: #4a5568;
        }.endpoint-row {
            display: flex; align-items: flex-start; gap: 1rem;
            padding: 0.9rem 1.4rem; border-bottom: 1px solid #1e2535;
            font-size: 0.875rem;
        }.endpoint-row:last-child { border-bottom: none; }.method {
            font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.72rem;
            font-weight: 700; padding: 0.2rem 0.55rem; border-radius: 5px;
            min-width: 46px; text-align: center; margin-top: 1px;
        }.get  { background: #1a3a2a; color: #68d391; }.post { background: #1a2a3a; color: #63b3ed; }.endpoint-path {
            font-family: 'SF Mono', 'Fira Code', monospace;
            color: #e2e8f0; font-size: 0.875rem; min-width: 210px;
        }.endpoint-desc { color: #718096; line-height: 1.5; }.code-block {
            background: #0d1117; border: 1px solid #2d3748;
            border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem;
        }.code-header {
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.65rem 1.2rem; background: #161b27;
            border-bottom: 1px solid #2d3748; font-size: 0.75rem;
            color: #4a5568; font-family: 'SF Mono', 'Fira Code', monospace;
        }
        pre {
            padding: 1.2rem 1.4rem;
            font-family: 'SF Mono', 'Fira Code', monospace;
            font-size: 0.8rem; line-height: 1.7; color: #a0aec0;
            overflow-x: auto; white-space: pre;
        }.kw  { color: #f6ad55; }.str { color: #68d391; }.cm  { color: #4a5568; font-style: italic; }.fn  { color: #63b3ed; }.num { color: #fc8181; }.links { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 2.5rem; }.link-btn {
            display: inline-flex; align-items: center; gap: 0.4rem;
            padding: 0.55rem 1.1rem; border-radius: 8px; font-size: 0.85rem;
            font-weight: 500; text-decoration: none; transition: opacity 0.15s;
        }.link-btn:hover { opacity: 0.8; }.link-primary  { background: #2b4c7e; color: #bee3f8; border: 1px solid #3182ce; }.link-secondary { background: #1a1f2e; color: #a0aec0; border: 1px solid #2d3748; }.data-schema {
            background: #0d1117; border: 1px solid #2d3748;
            border-radius: 12px; overflow: hidden; margin-bottom: 1.5rem;
        }.privacy-box {
            background: #0d1f17; border: 1px solid #1a4731;
            border-radius: 12px; padding: 1.2rem 1.6rem;
            margin-bottom: 1.5rem; font-size: 0.88rem;
            color: #68d391; line-height: 1.8;
        }.privacy-box span { color: #4a5568; }
        footer {
            margin-top: 1rem; font-size: 0.78rem;
            color: #2d3748; text-align: center; line-height: 1.8;
        }
        footer a { color: #2d3748; text-decoration: none; }
        footer a:hover { color: #4a5568; }
    </style>
</head>
<body>
<div class="container">

    <h1>UnifiedEdu</h1>
    <p class="subtitle">Federated Quiz Generation Aggregation Server</p>
    <p class="affiliation">
        Furkan Pala &amp; Islem Rekik  · 
        <a href="https://basira-lab.com" target="_blank">BASIRA Lab</a>,
        Imperial-X &amp; Department of Computing,
        Imperial College London
         · 
        <a href="mailto:f.pala23@imperial.ac.uk">f.pala23@imperial.ac.uk</a>
    </p>

    <div class="status-bar">
        <div class="dot"></div>
        Server is online and accepting federated updates
    </div>

    <!-- About -->
    <p class="section-title">About</p>
    <div class="description">
        This is the central aggregation server for <strong>UnifiedEdu</strong>,
        a privacy-preserving federated learning framework for automated educational
        quiz generation. Collaborating institutions train a quiz generation model
        <strong>locally on their own data</strong> and submit only a compact
        <strong>GNN parameter vector</strong> to this server. The server aggregates
        updates via <strong>weighted FedAvg</strong> and returns an improved global
        model — your raw data and full model weights
        <strong>never leave your machine</strong>.
        The framework is fully architecture-agnostic: BERT, LLaMA, T5, Qwen,
        or any custom PyTorch model is supported via a shared graph-based
        representation space.
    </div>

    <!-- Endpoints -->
    <p class="section-title">API Endpoints</p>
    <div class="endpoints">
        <div class="endpoint-header">Available Routes</div>
        <div class="endpoint-row">
            <span class="method get">GET</span>
            <span class="endpoint-path">/</span>
            <span class="endpoint-desc">This page — framework overview and usage guide</span>
        </div>
        <div class="endpoint-row">
            <span class="method get">GET</span>
            <span class="endpoint-path">/health</span>
            <span class="endpoint-desc">Server health check, current round, and mode status</span>
        </div>
        <div class="endpoint-row">
            <span class="method get">GET</span>
            <span class="endpoint-path">/get_global_parameters</span>
            <span class="endpoint-desc">Download the latest global GNN parameters before local training</span>
        </div>
        <div class="endpoint-row">
            <span class="method post">POST</span>
            <span class="endpoint-path">/aggregate</span>
            <span class="endpoint-desc">Submit your local GNN update — receive the aggregated global model in return</span>
        </div>
        <div class="endpoint-row">
            <span class="method get">GET</span>
            <span class="endpoint-path">/round_status</span>
            <span class="endpoint-desc">Check how many clients have submitted in the current round</span>
        </div>
    </div>

    <!-- POST /aggregate schema -->
    <p class="section-title">POST /aggregate — Payload Schema</p>
    <div class="data-schema">
        <div class="code-header">json</div>
<pre>{
  <span class="str">"client_id"</span>:   <span class="str">"your_institution_name"</span>,  <span class="cm">// string — unique ID given to you</span>
  <span class="str">"parameters"</span>:  [<span class="num">0.012</span>, <span class="num">-0.034</span>,...],    <span class="cm">// float array — flat GNN param vector</span>
  <span class="str">"num_samples"</span>: <span class="num">1200</span>,                    <span class="cm">// int — your local dataset size</span>
  <span class="str">"round"</span>:       <span class="num">3</span>                        <span class="cm">// int — current round number</span>
}</pre>
    </div>

    <!-- Quick start -->
    <p class="section-title">Quick Start</p>
    <div class="code-block">
        <div class="code-header">bash  ·  step 1 — install</div>
<pre>git clone https://github.com/furkanpala/UnifiedEdu
cd UnifiedEdu
pip install -r requirements.txt</pre>
    </div>

    <div class="code-block">
        <div class="code-header">python  ·  step 2 — run federated rounds</div>
<pre><span class="kw">from</span> unified_client <span class="kw">import</span> UnifiedClient
<span class="kw">from</span> transformers <span class="kw">import</span> AutoModel

<span class="cm"># Load your local model — stays on your machine</span>
model = AutoModel.<span class="fn">from_pretrained</span>(<span class="str">"bert-base-uncased"</span>)

<span class="cm"># Wrap it with UnifiedClient</span>
client = <span class="fn">UnifiedClient</span>(
    client_id   = <span class="str">"your_institution"</span>,
    server_url  = <span class="str">"https://unifiededu.onrender.com"</span>,
    model       = model,
    data_path   = <span class="str">"your_data.jsonl"</span>,
    num_samples = <span class="num">1200</span>,
)

<span class="cm"># Run federated rounds — one line per round</span>
<span class="kw">for</span> round_idx <span class="kw">in</span> <span class="fn">range</span>(<span class="num">10</span>):
    result = client.<span class="fn">run_round</span>()
    <span class="fn">print</span>(<span class="str">f"Round {result['round']} | Loss: {result['avg_loss']:.4f}"</span>)</pre>
    </div>

    <!-- Data format -->
    <p class="section-title">Data Format (.jsonl)</p>
    <div class="data-schema">
        <div class="code-header">json  ·  one object per line</div>
<pre>{
  <span class="str">"input_index"</span>:   <span class="num">1</span>,
  <span class="str">"clean_context"</span>: <span class="str">"Backpropagation is an algorithm used to train..."</span>,
  <span class="str">"qa_pairs"</span>: [
    {
      <span class="str">"question"</span>:   <span class="str">"What is backpropagation used for?"</span>,
      <span class="str">"answer"</span>:     <span class="str">"Training neural networks by computing gradients."</span>,
      <span class="str">"difficulty"</span>: <span class="str">"medium"</span>
    }
  ],
  <span class="str">"input_meta"</span>: {
    <span class="str">"title"</span>:      <span class="str">"Deep Learning Fundamentals"</span>,
    <span class="str">"authors"</span>:    [<span class="str">"J. Doe"</span>],
    <span class="str">"source"</span>:     <span class="str">"MIT OpenCourseWare"</span>,
    <span class="str">"categories"</span>: [<span class="str">"machine learning"</span>, <span class="str">"optimization"</span>]
  }
}</pre>
    </div>

    <!-- Links -->
    <p class="section-title">Resources</p>
    <div class="links">
        <a class="link-btn link-primary"
           href="https://github.com/furkanpala/UnifiedEdu" target="_blank">
            ⭐ GitHub Repository
        </a>
        <a class="link-btn link-secondary" href="/health" target="_blank">
            🔍 Health Check
        </a>
        <a class="link-btn link-secondary" href="/round_status" target="_blank">
            📊 Round Status
        </a>
        <a class="link-btn link-secondary"
           href="mailto:f.pala23@imperial.ac.uk">
            ✉️ Request Access
        </a>
    </div>

    <footer>
        UnifiedEdu  ·  BASIRA Lab  · 
        Imperial College London<br>
        <a href="http://arxiv.org/abs/2510.26350" target="_blank">
            arXiv 2510.26350
        </a>
         · 
        <a href="https://doi.org/10.1007/978-3-032-07904-6_4" target="_blank">
            LNCS 2026
        </a>
         · 
        <a href="https://github.com/microsoft/LoRA" target="_blank">
            LoRA
        </a>
    </footer>

</div>
</body>
</html>
""", 200

# ══════════════════════════════════════════════════════════════════════════════
# Routes — require FULL_MODE
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/get_global_parameters", methods=["GET"])
@requires_full_mode
def get_global_parameters():
    with aggregation_lock:
        flat         = flatten_params(global_gnn.state_dict())
        param_shapes = {k: list(v.shape) for k, v in global_gnn.state_dict().items()}
    return jsonify({
        "round":        current_round,
        "parameters":   flat,
        "param_shapes": param_shapes,
        "num_params":   len(flat),
    }), 200


@app.route("/aggregate", methods=["POST"])
@requires_full_mode
def aggregate():
    data = request.get_json(force=True)

    for field in ["client_id", "parameters", "num_samples"]:
        if field not in data:
            return jsonify({"error": f"Missing field: '{field}'"}), 400

    client_id    = str(data["client_id"])
    parameters   = data["parameters"]
    num_samples  = int(data["num_samples"])
    client_round = int(data.get("round", current_round))

    expected_len = sum(v.numel() for v in global_gnn.state_dict().values())
    if len(parameters) != expected_len:
        return jsonify({
            "error": f"Expected {expected_len} params, got {len(parameters)}."
        }), 400

    logger.info(f"Update from '{client_id}' | samples: {num_samples}")

    with aggregation_lock:
        pending_updates[client_id] = {
            "parameters":  parameters,
            "num_samples": num_samples,
            "timestamp":   time.time(),
        }
        num_pending = len(pending_updates)

    aggregated_this_call = False
    if num_pending >= MIN_CLIENTS_TO_AGGREGATE:
        run_aggregation()
        aggregated_this_call = True

    with aggregation_lock:
        global_flat = flatten_params(global_gnn.state_dict())

    return jsonify({
        "status":            "aggregated" if aggregated_this_call else "received",
        "round":             current_round,
        "global_parameters": global_flat,
        "message": (
            f"Aggregation complete. Round {current_round}."
            if aggregated_this_call else
            f"Waiting for {MIN_CLIENTS_TO_AGGREGATE - num_pending} more client(s)."
        ),
    }), 200


@app.route("/round_status", methods=["GET"])
@requires_full_mode
def round_status():
    with aggregation_lock:
        return jsonify({
            "current_round":   current_round,
            "pending_clients": list(pending_updates.keys()),
            "clients_needed":  max(0, MIN_CLIENTS_TO_AGGREGATE - len(pending_updates)),
            "round_elapsed_s": int(time.time() - round_start_time),
            "timeout_s":       ROUND_TIMEOUT_SECONDS,
        }), 200


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting UnifiedEdu server on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
