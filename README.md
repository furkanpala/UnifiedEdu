# UnifiedEdu 🎓

**Collaborative, privacy-preserving quiz generation via federated learning.**

Furkan Pala & Islem Rekik — BASIRA Lab, Imperial-X, Imperial College London

---

## What is it?

UnifiedEdu lets institutions collaboratively train a quiz generation model
**without sharing raw data or model weights.** Each collaborator trains
locally and sends only a small GNN parameter vector to the central server.
The server aggregates updates via FedAvg and returns an improved global model.

Works with **any PyTorch model** — BERT, LLaMA, T5, Qwen, or your own.

---

## Quick Start (Collaborators)

```bash
git clone https://github.com/furkanpala/UnifiedEdu
cd UnifiedEdu
pip install -r requirements.txt
```

```python
from unified_client import UnifiedClient
from transformers import AutoModel

model  = AutoModel.from_pretrained("bert-base-uncased")
client = UnifiedClient(
    client_id   = "your_institution",
    server_url  = "http://<SERVER_IP>:5000",  # contact us for address
    model       = model,
    data_path   = "your_data.jsonl",
    num_samples = 1200,
)

for round_idx in range(10):
    result = client.run_round()
    print(f"Round {result['round']} | Loss: {result['avg_loss']:.4f}")
```

## Data Format
```json
{
  "input_index": 1,
  "clean_context": "Your educational text here...",
  "qa_pairs": [{"question": "...", "answer": "...", "difficulty": "medium"}],
  "input_meta": {"title": "...", "source": "...", "categories": ["..."]}
}
```

## Server API

| Endpoint               | Method | Description                          |
|-----------------------|--------|--------------------------------------|
| `/health`             | GET    | Server status                        |
| `/get_global_parameters` | GET    | Download global GNN                 |
| `/aggregate`          | POST   | Submit update, receive global model  |
| `/round_status`       | GET    | Check round progress                 |

## Join the Federation
Email f.pala23@imperial.ac.uk with your institution name, model choice, and dataset size estimate. We will send you your client_id and server address.

