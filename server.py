# server.py
"""
UnifiedEdu Central Aggregation Server
======================================
Usage:
    python server.py

Dependencies:
    pip install flask numpy torch
"""

import os
import time
import copy
import threading
import logging
from typing import Dict, Any

import torch
import numpy as np
from flask import Flask, request, jsonify

from gnn.model import ModelGraphGNN
from config import GNNConfig

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

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Global State ───────────────────────────────────────────────────────────────
gnn_cfg = GNNConfig()
global_gnn = ModelGraphGNN(gnn_cfg)
global_gnn.eval()

# Thread-safe lock for aggregation
aggregation_lock = threading.Lock()

# Buffer: stores pending client updates before aggregation
# Format: { client_id: {"parameters": [...], "num_samples": int, "timestamp": float} }
pending_updates: Dict[str, Dict[str, Any]] = {}

# Aggregation config
MIN_CLIENTS_TO_AGGREGATE = int(os.environ.get("MIN_CLIENTS", 2))
ROUND_TIMEOUT_SECONDS    = int(os.environ.get("ROUND_TIMEOUT", 300))  # 5 min
current_round            = 0
round_start_time         = time.time()

# ── Helper: Flatten & Unflatten GNN params ─────────────────────────────────────

def flatten_params(state_dict: dict) -> list:
    """Flatten all GNN parameters into a single list of floats."""
    return torch.cat([v.cpu().float().flatten() for v in state_dict.values()]).tolist()


def unflatten_params(flat: list, reference_state_dict: dict) -> dict:
    """Reconstruct a state_dict from a flat list using reference shapes."""
    flat_tensor = torch.tensor(flat, dtype=torch.float32)
    new_state = {}
    offset = 0
    for key, ref_val in reference_state_dict.items():
        numel = ref_val.numel()
        new_state[key] = flat_tensor[offset: offset + numel].reshape(ref_val.shape)
        offset += numel
    return new_state


def fedavg(updates: Dict[str, Dict]) -> dict:
    """
    Weighted FedAvg aggregation.
    Each client's contribution is weighted by its num_samples.
    """
    total_samples = sum(u["num_samples"] for u in updates.values())
    ref_state     = global_gnn.state_dict()
    aggregated    = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in ref_state.items()}

    for client_id, update in updates.items():
        weight      = update["num_samples"] / total_samples
        client_dict = unflatten_params(update["parameters"], ref_state)
        for key in aggregated:
            aggregated[key] += weight * client_dict[key]

    logger.info(
        f"FedAvg over {len(updates)} clients | "
        f"total samples: {total_samples}"
    )
    return aggregated


def run_aggregation():
    """
    Aggregate all pending updates into the global GNN.
    Called automatically when enough clients have submitted.
    """
    global global_gnn, current_round, round_start_time, pending_updates

    with aggregation_lock:
        if len(pending_updates) < MIN_CLIENTS_TO_AGGREGATE:
            logger.warning(
                f"Aggregation triggered but only {len(pending_updates)} "
                f"update(s) available (min={MIN_CLIENTS_TO_AGGREGATE}). Skipping."
            )
            return

        logger.info(
            f"=== Starting aggregation for Round {current_round + 1} "
            f"with {len(pending_updates)} client(s) ==="
        )

        aggregated_params = fedavg(pending_updates)
        global_gnn.load_state_dict(aggregated_params)

        # Save checkpoint
        ckpt_path = f"checkpoints/global_gnn_round_{current_round + 1}.pt"
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(global_gnn.state_dict(), ckpt_path)
        logger.info(f"Global GNN checkpoint saved → {ckpt_path}")

        current_round    += 1
        round_start_time  = time.time()
        pending_updates   = {}  # Clear buffer for next round

        logger.info(f"=== Round {current_round} complete. Global model updated. ===")


# ── Background timeout aggregation ────────────────────────────────────────────

def timeout_aggregation_loop():
    """
    Background thread: triggers aggregation if round timeout is reached,
    even if MIN_CLIENTS haven't all responded yet.
    """
    while True:
        time.sleep(30)  # Check every 30 seconds
        elapsed = time.time() - round_start_time
        with aggregation_lock:
            has_updates = len(pending_updates) > 0
        if has_updates and elapsed > ROUND_TIMEOUT_SECONDS:
            logger.info(
                f"Round timeout reached ({ROUND_TIMEOUT_SECONDS}s). "
                f"Forcing aggregation with {len(pending_updates)} client(s)."
            )
            run_aggregation()


timeout_thread = threading.Thread(target=timeout_aggregation_loop, daemon=True)
timeout_thread.start()


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status":         "ok",
        "current_round":  current_round,
        "pending_clients": list(pending_updates.keys()),
        "min_clients":    MIN_CLIENTS_TO_AGGREGATE,
    }), 200


@app.route("/get_global_parameters", methods=["GET"])
def get_global_parameters():
    """
    Collaborators call this to download the latest global GNN parameters
    before starting their local training round.
    """
    with aggregation_lock:
        flat = flatten_params(global_gnn.state_dict())
        param_shapes = {
            k: list(v.shape)
            for k, v in global_gnn.state_dict().items()
        }

    logger.info(f"Global parameters requested. Round: {current_round}")
    return jsonify({
        "round":         current_round,
        "parameters":    flat,
        "param_shapes":  param_shapes,
        "num_params":    len(flat),
    }), 200


@app.route("/aggregate", methods=["POST"])
def aggregate():
    """
    Main endpoint. Collaborators POST their local GNN update here.

    Expected JSON payload:
    {
        "client_id":   "institution_name",
        "parameters":  [float, float,...],   ← flat GNN param vector
        "num_samples": 1200,                  ← local dataset size (for weighting)
        "round":       3                      ← which round this update is for
    }

    Returns:
    {
        "status":            "received" | "aggregated",
        "round":             int,
        "global_parameters": [float,...],    ← updated global params
        "message":           str
    }
    """
    data = request.get_json(force=True)

    # ── Validate payload ───────────────────────────────────────────────────────
    required_fields = ["client_id", "parameters", "num_samples"]
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: '{field}'"}), 400

    client_id   = str(data["client_id"])
    parameters  = data["parameters"]
    num_samples = int(data["num_samples"])
    client_round = int(data.get("round", current_round))

    # Validate parameter vector length
    expected_len = sum(v.numel() for v in global_gnn.state_dict().values())
    if len(parameters) != expected_len:
        return jsonify({
            "error": (
                f"Parameter vector length mismatch. "
                f"Expected {expected_len}, got {len(parameters)}."
            )
        }), 400

    logger.info(
        f"Received update from '{client_id}' | "
        f"samples: {num_samples} | round: {client_round}"
    )

    # ── Store update ───────────────────────────────────────────────────────────
    with aggregation_lock:
        pending_updates[client_id] = {
            "parameters":  parameters,
            "num_samples": num_samples,
            "timestamp":   time.time(),
        }
        num_pending = len(pending_updates)

    # ── Trigger aggregation if enough clients have submitted ───────────────────
    aggregated_this_call = False
    if num_pending >= MIN_CLIENTS_TO_AGGREGATE:
        run_aggregation()
        aggregated_this_call = True

    # ── Return updated global parameters ──────────────────────────────────────
    with aggregation_lock:
        global_flat = flatten_params(global_gnn.state_dict())

    status  = "aggregated" if aggregated_this_call else "received"
    message = (
        f"Aggregation complete. Global model updated to round {current_round}."
        if aggregated_this_call
        else (
            f"Update received. Waiting for "
            f"{MIN_CLIENTS_TO_AGGREGATE - num_pending} more client(s) "
            f"before aggregation."
        )
    )

    return jsonify({
        "status":            status,
        "round":             current_round,
        "global_parameters": global_flat,
        "message":           message,
    }), 200


@app.route("/round_status", methods=["GET"])
def round_status():
    """Returns current round status and which clients have submitted."""
    with aggregation_lock:
        return jsonify({
            "current_round":   current_round,
            "pending_clients": list(pending_updates.keys()),
            "clients_needed":  max(0, MIN_CLIENTS_TO_AGGREGATE - len(pending_updates)),
            "round_elapsed_s": int(time.time() - round_start_time),
            "timeout_s":       ROUND_TIMEOUT_SECONDS,
        }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))   # ← Cloud platforms set $PORT
    logger.info("=" * 60)
    logger.info("  UnifiedEdu Aggregation Server")
    logger.info(f"  Listening on port      : {port}")
    logger.info(f"  Min clients to aggregate : {MIN_CLIENTS_TO_AGGREGATE}")
    logger.info(f"  Round timeout            : {ROUND_TIMEOUT_SECONDS}s")
    logger.info("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False)
