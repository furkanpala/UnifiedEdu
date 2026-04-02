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
    """Landing page — always available."""
    #... (keep your existing landing page HTML here, unchanged)
    return "UnifiedEdu Server is running.", 200


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
