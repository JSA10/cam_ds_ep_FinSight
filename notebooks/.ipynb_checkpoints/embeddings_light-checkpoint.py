from functools import lru_cache
import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"  # or "BAAI/bge-small-en-v1.5"

@lru_cache(maxsize=1)
def _load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModel.from_pretrained(MODEL_ID)
    mdl.eval()
    return tok, mdl

@torch.inference_mode()
def encode(texts, batch_size=64):
    tok, mdl = _load()
    outs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i+batch_size], padding=True, truncation=True, return_tensors="pt")
        hs = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).expand(hs.size())
        pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        outs.append(F.normalize(pooled, p=2, dim=1))
    return torch.cat(outs, 0)
