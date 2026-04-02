# unified_client.py
"""
UnifiedEdu — Unified Client
============================

Minimal usage example:
-----------------------
    from unified_client import UnifiedClient

    client = UnifiedClient(
        client_id   = "my_institution",
        server_url  = "http://<SERVER_IP>:5000",
        model       = my_torch_model,          # any nn.Module
        data_path   = "my_data.jsonl",
        num_samples = 1200,
    )

    for round_idx in range(10):
        client.pull_global_parameters()        # 1. Get latest global GNN
        client.train_local_epoch()             # 2. Train locally
        response = client.push_local_update()  # 3. Send GNN params to server
        client.apply_global_update(response)   # 4. Apply aggregated result
"""

import copy
import json
import logging
from pathlib import Path
from typing import Optional, Union

import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

# ── Internal framework imports ─────────────────────────────────────────────────
from model_graph.converter import model_to_graph
from gnn.model import ModelGraphGNN
from quiz.head import QuizGenerationHead
from config import GNNConfig, QuizHeadConfig, DataConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
# Internal Dataset
# ══════════════════════════════════════════════════════════════════════════════

class _UnifiedEduDataset(Dataset):
    """
    Internal dataset loader for the UnifiedEdu JSONL format.
    Handles the collaborator's local.jsonl data file.
    """

    def __init__(self, data_path: str, tokenizer, cfg: DataConfig):
        self.tokenizer = tokenizer
        self.cfg       = cfg
        self.samples   = self._load(data_path)

    def _load(self, path: str):
        samples = []
        p = Path(path)
        if not p.exists():
            logger.warning(f"Data file not found: {path}. Using dummy data.")
            return self._dummy(20)

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                context = obj.get("clean_context", "")
                for qa in obj.get("qa_pairs", []):
                    samples.append({
                        "context":  context,
                        "question": qa.get("question", ""),
                        "answer":   qa.get("answer", ""),
                    })
        logger.info(f"Loaded {len(samples)} QA pairs from {path}")
        return samples

    def _dummy(self, n: int):
        return [
            {
                "context":  f"This is a sample educational context number {i}.",
                "question": f"What is context {i} about?",
                "answer":   f"It is about sample educational content {i}.",
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        tok = self.tokenizer

        def enc(text, max_len):
            return tok(
                text,
                max_length=max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        ctx = enc(s["context"],  self.cfg.max_context_len)
        q   = enc(s["question"], 64)
        a   = enc(s["answer"],   128)

        return {
            "context_ids":   ctx["input_ids"].squeeze(0),
            "context_mask":  ctx["attention_mask"].squeeze(0),
            "question_ids":  q["input_ids"].squeeze(0),
            "answer_ids":    a["input_ids"].squeeze(0),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Unified Client
# ══════════════════════════════════════════════════════════════════════════════

class UnifiedClient:
    """
    UnifiedEdu federated learning client.

    Wraps any PyTorch nn.Module and handles:
      - Model-graph conversion  (architecture → DAG)
      - Local GNN training      (only GNN params are learnable)
      - Communication with the  UnifiedEdu aggregation server
      - Privacy guarantee:      raw data & full model weights stay local

    Parameters
    ----------
    client_id   : str   — Unique identifier for your institution.
    server_url  : str   — Base URL of the UnifiedEdu server,
                          e.g. "http://123.45.67.89:5000"
    model       : nn.Module — Your local PyTorch model (LLM, BERT, etc.)
    data_path   : str   — Path to your local.jsonl data file.
    num_samples : int   — Number of training samples (used for FedAvg weighting).
    tokenizer_name : str — HuggingFace tokenizer identifier.
    device      : str   — "cuda" or "cpu".
    local_epochs : int  — Number of local training epochs per round.
    local_lr    : float — Learning rate for local GNN training.
    batch_size  : int   — Local training batch size.
    """

    def __init__(
        self,
        client_id:      str,
        server_url:     str,
        model:          nn.Module,
        data_path:      str,
        num_samples:    int,
        tokenizer_name: str  = "gpt2",
        device:         str  = "cpu",
        local_epochs:   int  = 2,
        local_lr:       float = 1e-4,
        batch_size:     int  = 4,
    ):
        self.client_id   = client_id
        self.server_url  = server_url.rstrip("/")
        self.model       = model
        self.num_samples = num_samples
        self.device      = torch.device(device)
        self.local_epochs = local_epochs
        self.local_lr    = local_lr
        self.current_round = 0

        # ── GNN & Quiz Head ────────────────────────────────────────────────────
        self.gnn_cfg  = GNNConfig()
        self.head_cfg = QuizHeadConfig()
        self.data_cfg = DataConfig()
        self.data_cfg.batch_size = batch_size

        self.gnn       = ModelGraphGNN(self.gnn_cfg).to(self.device)
        self.quiz_head = QuizGenerationHead(self.head_cfg).to(self.device)

        # ── Convert model to graph (done once — architecture is static) ────────
        logger.info(f"[{client_id}] Converting model architecture to graph...")
        self.model_graph: Data = model_to_graph(model, max_nodes=512).to(self.device)
        logger.info(
            f"[{client_id}] Model-graph ready: "
            f"{self.model_graph.num_nodes} nodes, "
            f"{self.model_graph.edge_index.size(1)} edges"
        )

        # ── Dataset & DataLoader ───────────────────────────────────────────────
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dataset = _UnifiedEduDataset(data_path, self.tokenizer, self.data_cfg)
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )
        # Override num_samples with actual dataset size if not provided
        if num_samples <= 0:
            self.num_samples = len(dataset)

        logger.info(
            f"[{client_id}] Client ready. "
            f"Dataset size: {self.num_samples} | Device: {device}"
        )

    # ── Parameter Utilities ────────────────────────────────────────────────────

    def _flatten_gnn_params(self) -> list:
        """Flatten GNN state_dict into a single list of floats."""
        return torch.cat([
            v.cpu().float().flatten()
            for v in self.gnn.state_dict().values()
        ]).tolist()

    def _unflatten_gnn_params(self, flat: list):
        """Load a flat parameter list back into the GNN."""
        flat_t  = torch.tensor(flat, dtype=torch.float32)
        new_sd  = {}
        offset  = 0
        ref_sd  = self.gnn.state_dict()
        for key, ref_val in ref_sd.items():
            numel = ref_val.numel()
            new_sd[key] = flat_t[offset: offset + numel].reshape(ref_val.shape)
            offset += numel
        self.gnn.load_state_dict(new_sd)

    # ── Server Communication ───────────────────────────────────────────────────

    def pull_global_parameters(self) -> bool:
        """
        Download the latest global GNN parameters from the server
        and apply them locally.

        Returns True on success, False on failure.
        """
        url = f"{self.server_url}/get_global_parameters"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            self._unflatten_gnn_params(data["parameters"])
            self.current_round = data["round"]
            logger.info(
                f"[{self.client_id}] Pulled global parameters. "
                f"Server round: {self.current_round}"
            )
            return True
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to pull global parameters: {e}")
            return False

    def push_local_update(self) -> Optional[dict]:
        """
        Send local GNN parameters to the server's /aggregate endpoint.

        Returns the server's JSON response (including updated global params),
        or None on failure.
        """
        url     = f"{self.server_url}/aggregate"
        payload = {
            "client_id":   self.client_id,
            "parameters":  self._flatten_gnn_params(),
            "num_samples": self.num_samples,
            "round":       self.current_round,
        }
        try:
            logger.info(
                f"[{self.client_id}] Sending GNN parameters to {url}..."
            )
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                f"[{self.client_id}] Server response: "
                f"status={result.get('status')} | "
                f"round={result.get('round')} | "
                f"{result.get('message', '')}"
            )
            return result
        except Exception as e:
            logger.error(f"[{self.client_id}] Failed to push update: {e}")
            return None

    def apply_global_update(self, server_response: dict):
        """
        Apply the aggregated global GNN parameters returned by the server.

        Parameters
        ----------
        server_response : dict — The JSON response from /aggregate.
        """
        global_params = server_response.get("global_parameters")
        if global_params is None:
            logger.warning(
                f"[{self.client_id}] No global_parameters in server response."
            )
            return
        self._unflatten_gnn_params(global_params)
        self.current_round = server_response.get("round", self.current_round)
        logger.info(
            f"[{self.client_id}] Applied global update. "
            f"Now at round {self.current_round}."
        )

    # ── Local Training ─────────────────────────────────────────────────────────

    def train_local_epoch(self) -> float:
        """
        Train the local GNN + quiz head for self.local_epochs epochs.
        The underlying model (LLM) is kept FROZEN — only GNN params update.

        Returns
        -------
        avg_loss : float
        """
        self.gnn.train()
        self.quiz_head.train()

        optimizer = optim.AdamW(
            list(self.gnn.parameters()) + list(self.quiz_head.parameters()),
            lr=self.local_lr,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_token_id or 0
        )

        total_loss, total_steps = 0.0, 0

        for epoch in range(self.local_epochs):
            for batch in self.dataloader:
                optimizer.zero_grad()

                context_ids  = batch["context_ids"].to(self.device)
                context_mask = batch["context_mask"].to(self.device)
                question_ids = batch["question_ids"].to(self.device)
                answer_ids   = batch["answer_ids"].to(self.device)

                B = context_ids.size(0)

                # Get graph embedding (shared for all samples in batch)
                graph_emb = self.gnn(self.model_graph).expand(B, -1)

                # Forward through quiz head
                outputs = self.quiz_head(
                    context_ids=context_ids,
                    context_mask=context_mask,
                    question_ids=question_ids[:, :-1],
                    answer_ids=answer_ids[:, :-1],
                    graph_embedding=graph_emb,
                )

                # Question loss
                q_logits = outputs["question_logits"]
                q_loss   = criterion(
                    q_logits.reshape(-1, q_logits.size(-1)),
                    question_ids[:, 1:].reshape(-1),
                )

                # Answer loss
                a_logits = outputs["answer_logits"]
                a_loss   = criterion(
                    a_logits.reshape(-1, a_logits.size(-1)),
                    answer_ids[:, 1:].reshape(-1),
                )

                loss = q_loss + a_loss
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.gnn.parameters()) + list(self.quiz_head.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()

                total_loss  += loss.item()
                total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        logger.info(
            f"[{self.client_id}] Local training done. "
            f"Avg loss: {avg_loss:.4f} | Steps: {total_steps}"
        )
        return avg_loss

    # ── Full Round (convenience method) ───────────────────────────────────────

    def run_round(self) -> dict:
        """
        Execute a complete federated round in one call:
          1. Pull global GNN parameters from server
          2. Train locally
          3. Push updated GNN parameters to server
          4. Apply aggregated global update

        Returns the server response dict.
        """
        logger.info(f"[{self.client_id}] ── Starting federated round ──")

        # Step 1: Pull latest global model
        self.pull_global_parameters()

        # Step 2: Local training
        avg_loss = self.train_local_epoch()

        # Step 3 & 4: Push and apply
        response = self.push_local_update()
        if response:
            self.apply_global_update(response)

        return {
            "round":    self.current_round,
            "avg_loss": avg_loss,
            "status":   response.get("status") if response else "failed",
        }

    # ── Quiz Generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_quiz(self, context: str, max_new_tokens: int = 50) -> dict:
        """
        Generate a quiz question and answer from a context string.

        Parameters
        ----------
        context        : str — The educational text to generate a quiz from.
        max_new_tokens : int — Maximum tokens to generate.

        Returns
        -------
        dict with keys "question" and "answer".
        """
        self.gnn.eval()
        self.quiz_head.eval()

        enc = self.tokenizer(
            context,
            max_length=self.data_cfg.max_context_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        context_ids  = enc["input_ids"].to(self.device)
        context_mask = enc["attention_mask"].to(self.device)
        graph_emb    = self.gnn(self.model_graph)
        memory       = self.quiz_head.encode_context(
            context_ids, context_mask, graph_emb
        )

        def greedy(mode):
            bos = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
            ids = torch.tensor([[bos]], device=self.device)
            for _ in range(max_new_tokens):
                logits = (
                    self.quiz_head.decode_question(ids, memory)
                    if mode == "question"
                    else self.quiz_head.decode_answer(ids, memory)
                )
                nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                ids = torch.cat([ids, nxt], dim=1)
                if nxt.item() == self.tokenizer.eos_token_id:
                    break
            return self.tokenizer.decode(ids[0], skip_special_tokens=True)

        return {
            "question": greedy("question"),
            "answer":   greedy("answer"),
        }
